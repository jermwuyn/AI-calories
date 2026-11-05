"""Validate prepared manifests and run bounded real-data CPU/GPU checks, never full training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import classification as trainer
from classification.unified import load_unified, open_sample, ArchiveReader
from classification.model import build_model, set_train_mode
import torch
import yaml
import nbformat


def validate():
    cfg = yaml.safe_load((ROOT / 'configs/classification-unified-exp001.yaml').read_text())
    root, mapping, data = load_unified(ROOT, cfg, ('train', 'val', 'test'))
    report = json.loads((root / 'report.json').read_text())
    sources = set(mapping['sources'])
    assert sources == {'30VNFoods', 'UEC-Food256', 'VinaFood21', 'Vietnamese_StreetFood_14Class'}
    rows = [r for ds in data.values() for r in ds.rows]
    assert set(r['source'] for r in rows) == sources
    assert len({r['sample_id'] for r in rows}) == len(rows)
    images = [json.loads(line) for line in (root / 'image_inventory.jsonl').read_text().splitlines()]
    seen, exact, native, groups = {}, defaultdict(set), defaultdict(set), defaultdict(set)
    keys = set()
    archives = {}
    for r in rows:
        im = images[r['image_index']]
        assert (r['archive'], r['member'], r['source']) == (im['archive'], im['member'], im['source'])
        assert 'dataset-v1' not in r['archive'] and '/detection/' not in r['member']
        exact[im['pixel_sha256']].add(r['split'])
        native[im['native_group']].add(r['split'])
        groups[r['group_id']].add(r['split'])
        key = (im['pixel_sha256'], tuple(r['bbox'] or ()), r['canonical_id'])
        assert key not in keys
        keys.add(key)
        if r['archive'] not in archives: archives[r['archive']] = zipfile.ZipFile(r['archive'])
        assert r['member'] in archives[r['archive']].NameToInfo
        if r['bbox']:
            x1, y1, x2, y2 = r['bbox']
            assert min(x1, y1) >= 0 and x2-x1 >= 16 and y2-y1 >= 16
        seen.setdefault((r['source'], r['source_class_id'], r['split']), r)
    for table in (exact, native, groups): assert all(len(v) == 1 for v in table.values())
    # Check every retained duplicate edge against the actual final membership.
    membership = defaultdict(set)
    for r in rows: membership[r['image_index']].add(r['split'])
    edge_count = 0
    for line in (root / 'duplicate_edges.jsonl').read_text().splitlines():
        edge = json.loads(line)
        assert len(membership[edge['a']] | membership[edge['b']]) <= 1
        edge_count += 1
    by_id = {r['sample_id']: r['split'] for r in rows}
    region_edges = 0
    for line in (root / 'region_duplicate_edges.jsonl').read_text().splitlines():
        edge = json.loads(line)
        assert len({by_id[s] for s in (edge['a'], edge['b']) if s in by_id}) <= 1
        region_edges += 1
    reader = ArchiveReader()
    for r in seen.values():
        image = open_sample(r, reader)
        assert min(image.size) >= 16
    nb = nbformat.read(ROOT / 'notebooks/EXP-001_UnifiedFood_4Datasets.ipynb', as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code': compile(cell.source, '<notebook>', 'exec')
    result = {'valid_samples': len(rows), 'splits': {s: len(ds) for s, ds in data.items()},
              'canonical_classes': len(mapping['classes']), 'source_class_split_crop_checks': len(seen),
              'checked_duplicate_edges': edge_count, 'exact_and_group_leakage': 0,
              'checked_region_duplicate_edges': region_edges,
              'notebook_schema_and_python_syntax': 'passed',
              'missing_train_classes': sorted(set(range(len(mapping['classes']))) - {r['canonical_id'] for r in data['train'].rows}),
              'all_original_images_decoded_during_prepare': True}
    assert result['splits'] == report['splits']
    result['missing_train_class_names'] = [mapping['classes'][i]['source_name'] for i in result['missing_train_classes']]
    (root / 'validation_report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return cfg, root, mapping, data


def smoke(cfg, root, mapping, data, device_name):
    device = torch.device(device_name)
    torch.set_num_threads(4)
    amp = device.type == 'cuda'
    ds = data['train']
    indices = [next(i for i, r in enumerate(ds.rows) if r['source'] == source) for source in mapping['sources']]
    samples = [ds[i] for i in indices]
    x = torch.stack([s[0] for s in samples]).to(device)
    y = torch.tensor([s[1] for s in samples], device=device)
    model = build_model(cfg['architecture'], len(mapping['classes']), True).to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    losses = {}
    skipped = {}
    def step(net, opt, scale):
        for attempt in range(12):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = net(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
            assert logits.shape == (4, len(mapping['classes'])) and torch.isfinite(loss)
            scale.scale(loss).backward(); scale.unscale_(opt)
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['gradient_clip'])
            scale.step(opt); scale.update()
            if torch.isfinite(norm): return loss, attempt
            assert amp, 'Non-finite CPU gradients'
        raise AssertionError('AMP did not recover finite gradients after scale backoff')
    for stage in ('head', 'finetune'):
        optimizer, scheduler = trainer.make_optim(model, cfg, stage)
        set_train_mode(model, stage)
        loss, skipped[stage] = step(model, optimizer, scaler)
        scheduler.step()
        losses[stage] = float(loss.detach())
    model.eval()
    with torch.inference_mode(): expected = model(x)
    vx, vy = data['val'][0]
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        validation_loss = torch.nn.functional.cross_entropy(model(vx.unsqueeze(0).to(device)), torch.tensor([vy], device=device))
    assert torch.isfinite(validation_loss)
    with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as tmp:
        checkpoint = Path(tmp) / 'smoke.pth'
        trainer.save_checkpoint(checkpoint, model, optimizer, scheduler, scaler, 0, 0., 0,
                                'finetune', cfg, mapping, root, [])
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        resumed = build_model(cfg['architecture'], len(mapping['classes']), False).to(device)
        resumed.load_state_dict(ck['model'])
        opt2, sched2 = trainer.make_optim(resumed, cfg, 'finetune')
        opt2.load_state_dict(ck['optimizer']); sched2.load_state_dict(ck['scheduler'])
        scaler2 = torch.amp.GradScaler('cuda', enabled=amp); scaler2.load_state_dict(ck['scaler'])
        trainer.restore_rng(ck['rng'])
        resumed.eval()
        with torch.inference_mode(): actual = resumed(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        before = resumed.classifier[-1].weight.detach().clone()
        set_train_mode(resumed, 'finetune')
        loss, skipped['resume'] = step(resumed, opt2, scaler2)
        sched2.step()
        assert not torch.equal(before, resumed.classifier[-1].weight)
    result = {'device': device_name, 'amp': amp, 'pretrained': True, 'sources_in_batch': mapping['sources'],
              'batch_size': 4, 'losses': losses, 'resumed_loss': float(loss.detach()),
              'validation_loss': float(validation_loss),
              'amp_backoff_steps': skipped,
              'checkpoint_restore_logits': 'passed', 'resumed_optimizer_step': 'passed',
              'parameters': sum(p.numel() for p in model.parameters()),
              'head_parameters': sum(p.numel() for p in model.classifier.parameters()), 'full_training_started': False}
    (root / f'smoke_{device_name}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg, root, mapping, data = validate()
    if args.smoke:
        smoke(cfg, root, mapping, data, 'cpu')
        if torch.cuda.is_available(): smoke(cfg, root, mapping, data, 'cuda')
