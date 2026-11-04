"""Train, evaluate, or smoke-test the independent classification heads."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault('TORCH_HOME', str(PROJECT / '.cache' / 'torch'))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from classification.data import ManifestDataset, load_spec
from classification.metrics import Metrics
from classification.model import build_model, set_stage, set_train_mode


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(dataset, cfg, split, epoch=0):
    generator = torch.Generator().manual_seed(cfg['seed'] + epoch)
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=(split == 'train'),
                      num_workers=cfg['workers'], pin_memory=torch.cuda.is_available(), generator=generator)


def load_datasets(cfg, splits):
    root, mapping = load_spec(PROJECT, cfg['dataset_config'], cfg['dataset'])
    datasets = {s: ManifestDataset(root, mapping, s) for s in splits}
    seen = set()
    for split, dataset in datasets.items():
        if seen.intersection(dataset.ids):
            raise ValueError(f'Image ID occurs in multiple splits: {split}')
        seen.update(dataset.ids)
    print('Dataset:', cfg['dataset'], {s: len(ds) for s, ds in datasets.items()}, flush=True)
    return root, mapping, datasets


def make_optim(model, cfg, stage):
    set_stage(model, stage)
    lr = cfg['head_lr'] if stage == 'head' else cfg['finetune_lr']
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr,
                                  weight_decay=cfg['weight_decay'])
    length = cfg['head_epochs'] if stage == 'head' else cfg['finetune_epochs']
    return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=length)


def run_epoch(model, loader, mapping, device, scaler, amp, optimizer=None, stage=None):
    training = optimizer is not None
    if training:
        set_train_mode(model, stage)
    else:
        model.eval()
    stats = Metrics(len(mapping['classes']))
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), loader.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        stats.update(logits, labels, loss)
    return stats.result(mapping['classes'])


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, bad_epochs, stage, cfg, mapping, root, history):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                'best_validation_macro_f1': best, 'bad_epochs': bad_epochs, 'stage': stage,
                'config': cfg, 'dataset': {'head': mapping['head'], 'class_map': mapping,
                                           'dataset_root': str(root), 'dataset_version': 'dataset-v1'},
                'history': history, 'rng': rng_state()}, path)


def train(cfg, output, resume):
    seed_all(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = bool(cfg['amp'] and device.type == 'cuda')
    root, mapping, data = load_datasets(cfg, ('train', 'val'))
    if resume:
        ckpt = torch.load(resume, map_location='cpu', weights_only=False)
        if ckpt['config'] != cfg or ckpt['dataset']['class_map'] != mapping or ckpt['dataset']['dataset_root'] != str(root):
            raise ValueError('Resume config, class map, or dataset root differs from checkpoint')
        if Path(resume).resolve().parent != output.resolve():
            raise ValueError('Resume output must be the checkpoint directory')
        model = build_model(cfg['architecture'], len(mapping['classes']), False).to(device)
        model.load_state_dict(ckpt['model'])
        start, best, bad, history, prev_stage = ckpt['epoch'] + 1, ckpt['best_validation_macro_f1'], ckpt['bad_epochs'], ckpt['history'], ckpt['stage']
    else:
        if output.exists():
            raise FileExistsError(f'Output already exists: {output}')
        model = build_model(cfg['architecture'], len(mapping['classes']), cfg['pretrained']).to(device)
        output.mkdir(parents=True)
        (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        (output / 'class_map.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf-8')
        start, best, bad, history, prev_stage = 0, -1.0, 0, [], None
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    optimizer = scheduler = None
    if resume:
        optimizer, scheduler = make_optim(model, cfg, prev_stage)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        restore_rng(ckpt['rng'])
    writer = SummaryWriter(str(output / 'tensorboard'))
    try:
        total_epochs = cfg['head_epochs'] + cfg['finetune_epochs']
        for epoch in range(start, total_epochs):
            stage = 'head' if epoch < cfg['head_epochs'] else 'finetune'
            if stage != prev_stage:
                optimizer, scheduler = make_optim(model, cfg, stage)
                bad = 0
                prev_stage = stage
            loader = make_loader(data['train'], cfg, 'train', epoch)
            loader.gradient_clip = cfg['gradient_clip']
            train_metrics = run_epoch(model, loader, mapping, device, scaler, amp, optimizer, stage)
            val_metrics = run_epoch(model, make_loader(data['val'], cfg, 'val'), mapping, device, scaler, amp)
            improved = val_metrics['macro_f1'] > best
            if improved:
                best, bad = val_metrics['macro_f1'], 0
            else:
                bad += 1
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            history.append({'epoch': epoch + 1, 'stage': stage, 'lr': lr, 'train': train_metrics, 'val': val_metrics})
            for split, metrics in (('train', train_metrics), ('val', val_metrics)):
                for key in ('loss', 'top1_accuracy', 'top3_accuracy', 'macro_f1'):
                    writer.add_scalar(f'{split}/{key}', metrics[key], epoch + 1)
            writer.add_scalar('learning_rate', lr, epoch + 1)
            writer.flush()
            args = (model, optimizer, scheduler, scaler, epoch, best, bad, stage, cfg, mapping, root, history)
            save_checkpoint(output / 'last.pth', *args)
            if improved:
                save_checkpoint(output / 'best.pth', *args)
            (output / 'metrics.json').write_text(json.dumps({'history': history, 'best_validation_macro_f1': best}, indent=2), encoding='utf-8')
            print(f"Epoch {epoch + 1}/{total_epochs} {stage}: train loss={train_metrics['loss']:.4f}, val loss={val_metrics['loss']:.4f}, val macro-F1={val_metrics['macro_f1']:.4f}", flush=True)
            if bad >= cfg['patience']:
                if stage == 'head' and cfg['finetune_epochs']:
                    # Continue directly to fine-tuning while preserving the best checkpoint.
                    print('Head stage reached patience; remaining head epochs will be skipped.', flush=True)
                    # Use a planned transition marker so resume follows the same schedule.
                    ck = torch.load(output / 'last.pth', map_location='cpu', weights_only=False)
                    ck['epoch'] = cfg['head_epochs'] - 1
                    torch.save(ck, output / 'last.pth')
                    # Skip via a fresh resume to avoid silently continuing head updates.
                    writer.close()
                    return train(cfg, output, output / 'last.pth')
                print('Early stopping.', flush=True)
                break
    finally:
        writer.close()


def evaluate(checkpoint, split, output):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    seed_all(cfg['seed'])
    root, mapping, data = load_datasets(cfg, (split,))
    if mapping != ck['dataset']['class_map'] or str(root) != ck['dataset']['dataset_root']:
        raise ValueError('Checkpoint mapping or dataset root differs')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg['architecture'], len(mapping['classes']), False).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    metrics = run_epoch(model, make_loader(data[split], cfg, split), mapping, device, scaler, False)
    sample = data[split][0][0].unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(10): model(sample)
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50): model(sample)
        if device.type == 'cuda': torch.cuda.synchronize()
    metrics['inference_latency_ms_batch1'] = (time.perf_counter() - start) * 1000 / 50
    metrics['latency_scope'] = 'model forward only, batch size 1, after 10 warmup runs; excludes preprocessing and transfer'
    metrics['checkpoint'] = str(Path(checkpoint).resolve())
    metrics['split'] = split
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: metrics[k] for k in ('split', 'loss', 'top1_accuracy', 'top3_accuracy', 'macro_f1', 'inference_latency_ms_batch1')}, indent=2))


def smoke(cfg):
    seed_all(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = bool(cfg['amp'] and device.type == 'cuda')
    root, mapping, data = load_datasets(cfg, ('train', 'val', 'test'))
    model = build_model(cfg['architecture'], len(mapping['classes']), cfg['pretrained']).to(device)
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    optimizer, scheduler = make_optim(model, cfg, 'head')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    set_train_mode(model, 'head')
    x, y = next(iter(make_loader(data['train'], cfg, 'train')))
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=amp):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
    scaler.step(optimizer)
    scaler.update()
    model.eval()
    vx, vy = next(iter(make_loader(data['val'], cfg, 'val')))
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        val_logits = model(vx.to(device))
        val_loss = nn.functional.cross_entropy(val_logits, vy.to(device))
    set_train_mode(model, 'finetune')
    optimizer, _ = make_optim(model, cfg, 'finetune')
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=amp):
        finetune_loss = nn.functional.cross_entropy(model(x), y)
    scaler.scale(finetune_loss).backward()
    scaler.unscale_(optimizer)
    finetune_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
    scaler.step(optimizer)
    scaler.update()
    if device.type == 'cuda': torch.cuda.synchronize()
    print(json.dumps({'dataset': cfg['dataset'], 'device': str(device), 'pretrained': cfg['pretrained'],
                      'total_parameters': sum(p.numel() for p in model.parameters()),
                      'head_trainable_parameters': sum(p.numel() for p in model.classifier.parameters()),
                      'finetune_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'input': list(x.shape), 'logits': list(logits.shape), 'validation_logits': list(val_logits.shape),
                      'loss': float(loss.detach()), 'validation_loss': float(val_loss), 'gradient_norm': float(norm),
                      'finetune_loss': float(finetune_loss.detach()), 'finetune_gradient_norm': float(finetune_norm),
                      'peak_allocated_mib': round(torch.cuda.max_memory_allocated() / 2**20, 2) if device.type == 'cuda' else None,
                      'peak_reserved_mib': round(torch.cuda.max_memory_reserved() / 2**20, 2) if device.type == 'cuda' else None}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'evaluate', 'smoke'])
    parser.add_argument('--config', default='configs/classification-exp001.yaml')
    parser.add_argument('--dataset', choices=['30vn', 'vina21'])
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    if args.command == 'evaluate':
        if not args.checkpoint or not args.output: parser.error('evaluate requires --checkpoint and --output')
        evaluate(args.checkpoint, args.split, args.output)
        return
    cfg = yaml.safe_load((PROJECT / args.config).read_text(encoding='utf-8'))
    for name in ('dataset', 'pretrained', 'batch_size', 'workers'):
        value = getattr(args, name)
        if value is not None: cfg[name] = value
    if cfg['head_epochs'] < 0 or cfg['finetune_epochs'] < 0 or cfg['head_epochs'] + cfg['finetune_epochs'] <= 0 or cfg['patience'] < 1 or cfg['batch_size'] < 1 or cfg['workers'] < 0:
        parser.error('Invalid epoch, patience, batch size, or worker configuration')
    if args.command == 'smoke': smoke(cfg)
    else:
        if not args.output: parser.error('train requires --output')
        train(cfg, args.output, args.resume)


if __name__ == '__main__':
    main()
