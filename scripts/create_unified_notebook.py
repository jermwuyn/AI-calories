"""Generate a reproducible, safe-by-default experiment notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# EXP-001 — Unified Food Classification: 4 original datasets

## 1. Project overview
30VNFoods + UEC-Food256 + VinaFood21 + Vietnamese_StreetFood_14Class.
Original archives are read-only. No Unified Detection data is loaded. Classification
images retain source labels; detection annotations yield individual crops, including
multiple crops per image. Ambiguous taxonomy matches remain separate.

Full training and test evaluation are **disabled by default**. The training cell calls
the existing trainer with every valid train sample and no batch limit.
''')
code('''from pathlib import Path
import sys, os, json
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
assert (ROOT / 'scripts/classification.py').is_file()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, yaml
from classification.unified import load_unified, open_sample, ArchiveReader
from classification.model import build_model
from classification.data import preprocess
from scripts import classification as trainer
CFG = yaml.safe_load((ROOT / 'configs/classification-unified-exp001.yaml').read_text())
DATA = ROOT / CFG['manifest_root']
RUN = ROOT / 'runs/exp-001-unified4'
RUN_FULL_TRAIN = False
RUN_TEST_EVALUATION = False
''')
md('## 2. Python and CUDA\nSelect the `ai-unified` kernel backed by `D:\\CondaEnvs\\ai\\python.exe`.')
code('''print('Python:', sys.executable, sys.version)
assert Path(sys.executable).resolve() == Path('D:/CondaEnvs/ai/python.exe').resolve()
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unavailable — CPU supported')
assert CFG['seed'] is None
''')
md('## 3. Four-source audit\nCounts refer to original archive entries. Identical images can appear in multiple class folders; consult duplicate audit for unique content. VinaFood21 has no native validation split; UEC splits are explicitly derived. No source validation/test samples enter train.')
code('''report = json.loads((DATA / 'report.json').read_text(encoding='utf-8'))
display(pd.DataFrame(report['sources']).T.fillna(0))
display(pd.DataFrame(report['samples_by_source']).T.fillna(0).astype(int))
print(report['split_policy'])
print(report['near_duplicate_limit'])
display(pd.Series(report['exclusions'], name='excluded'))
''')
md('## 4. Unified class mapping\nOnly exact labels normalized for case and separators are merged; semantic translations require review. Source IDs and canonical IDs are recorded independently.')
code('''root, mapping, datasets = load_unified(ROOT, CFG, ('train', 'val', 'test'))
display(pd.DataFrame(mapping['provenance']))
print('Canonical classes:', len(mapping['classes']))
print('Merged source classes:', report['merged_class_count'])
display(pd.read_csv(DATA / 'taxonomy_review.csv').head(20))
''')
md('## 5. Class distribution\n\nAfter leakage quarantine, classes 307 (Hue-style Spicy Beef Noodle Soup) and 310 (Grilled Pork Vermicelli) have no retained train samples. Their output classes remain, but cannot learn positive examples. No holdout rows are moved into train. VinaFood21 has no native validation split.')
code('''rows = pd.DataFrame([r for ds in datasets.values() for r in ds.rows])
distribution = pd.crosstab(rows.canonical_id, rows.split).reindex(range(len(mapping['classes'])), fill_value=0)
distribution.plot.bar(stacked=True, figsize=(20, 5), width=1)
plt.xticks(range(0, len(mapping['classes']), 10)); plt.title('Valid samples by canonical ID'); plt.tight_layout(); plt.show()
print('Classes without training support:', distribution.index[distribution.train.eq(0)].tolist())
''')
md('## 6. Original images and detection crops\nEach detection sample retains its source annotation and original image identity.')
code('''reader = ArchiveReader()
fig, axes = plt.subplots(4, 2, figsize=(10, 15))
for ax, source in zip(axes, mapping['sources']):
    row = next(r for ds in datasets.values() for r in ds.rows if r['source'] == source)
    ax[0].imshow(open_sample(dict(row, bbox=None), reader)); ax[0].set_title(source + ' — original')
    ax[1].imshow(open_sample(row, reader)); ax[1].set_title(mapping['classes'][row['canonical_id']]['source_name'])
    for a in ax: a.axis('off')
plt.tight_layout(); plt.show()
''')
md('## 7. Augmentation preview\nRandomResizedCrop(224, scale 0.8–1.0), horizontal flip, ImageNet normalization; no fixed seed.')
code('''sample = open_sample(datasets['train'].rows[0], reader)
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax in axes:
    x = preprocess(True)(sample).permute(1, 2, 0).numpy()
    x = np.clip(x * np.array([.229, .224, .225]) + np.array([.485, .456, .406]), 0, 1)
    ax.imshow(x); ax.axis('off')
plt.tight_layout(); plt.show()
''')
md('## 8. EfficientNet-B0 architecture and parameter count\nPretrained backbone, dynamic linear head; head warm-up followed by full fine-tuning.')
code('''model = build_model(CFG['architecture'], len(mapping['classes']), CFG['pretrained'])
print(model)
print('Total parameters:', sum(p.numel() for p in model.parameters()))
print('Head parameters:', sum(p.numel() for p in model.classifier.parameters()))
''')
md('## 9. Full training using the existing trainer\nAdamW + CrossEntropy + cosine schedule, AMP on CUDA, clipping, TensorBoard, best/last checkpoints, RNG-restoring resume and validation Macro-F1 early stopping. Set RUN_FULL_TRAIN=True only when ready. All eligible train rows are loaded each epoch. Do not select epochs using test results.')
code('''RESUME = None  # e.g. RUN / 'last.pth'; keep the same config and output directory
if RUN_FULL_TRAIN:
    trainer.train(CFG, RUN, RESUME)
else:
    print('Full training has not been started. Train samples:', len(datasets['train']))
print('TensorBoard: python -m tensorboard.main --logdir runs/exp-001-unified4/tensorboard')
''')
md('## 10. Train/validation loss, accuracy and Macro-F1')
code('''if (RUN / 'metrics.json').exists():
    history = json.loads((RUN / 'metrics.json').read_text())['history']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ['loss', 'top1_accuracy', 'macro_f1']):
        for split in ('train', 'val'):
            ax.plot([h['epoch'] for h in history], [h[split][key] for h in history], label=split)
        ax.set_title(key); ax.set_xlabel('Epoch'); ax.legend()
    plt.tight_layout(); plt.show()
else:
    print('No training history yet; no fabricated metrics.')
''')
md('## 11. Confusion matrix\nValidation matrix for the epoch selected by validation Macro-F1. Full-resolution values remain in metrics.json.')
code('''if (RUN / 'metrics.json').exists():
    best = max(history, key=lambda h: h['val']['macro_f1'])
    matrix = np.asarray(best['val']['confusion_matrix'])
    plt.figure(figsize=(10, 8)); plt.imshow(np.log1p(matrix), cmap='Blues'); plt.colorbar(label='log(1 + count)')
    plt.xlabel('Predicted canonical ID'); plt.ylabel('True canonical ID'); plt.title('Validation confusion matrix'); plt.show()
else:
    print('Available after training.')
''')
md('## 12. Held-out test evaluation after training\nEnable once the validation-selected best checkpoint is final. Evaluation uses every retained test sample; its output is not overwritten.')
code('''test_path = RUN / 'test_metrics.json'
if RUN_TEST_EVALUATION:
    assert (RUN / 'best.pth').exists(), 'Train first'
    trainer.evaluate(RUN / 'best.pth', 'test', test_path)
if test_path.exists():
    test_metrics = json.loads(test_path.read_text())
    display({k: test_metrics[k] for k in ['loss', 'top1_accuracy', 'top3_accuracy', 'macro_f1']})
    display(pd.DataFrame(test_metrics['per_class']))
else:
    print('Test has not been evaluated.')
''')
md('## 13. Example inference\nRuns only with the trained checkpoint; uses a validation crop for illustration.')
code('''if (RUN / 'best.pth').exists():
    ck = torch.load(RUN / 'best.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model']); model.eval()
    x, target = datasets['val'][0]
    with torch.inference_mode(): probabilities = model(x.unsqueeze(0)).softmax(1)[0]
    values, ids = probabilities.topk(5)
    display(pd.DataFrame({'class': [mapping['classes'][i]['source_name'] for i in ids.tolist()], 'probability': values.tolist()}))
    print('Target:', mapping['classes'][target]['source_name'])
else:
    print('Inference awaits a trained checkpoint.')
''')
md('''## 14. Conclusions
The experiment uses exactly four original sources and an auditable conservative taxonomy.
The manifest references original ZIP entries and creates crops at read time; only nested
ZIP containers are cached. Exclusions and duplicate edges are retained for inspection.
Near-duplicate checks are heuristic and cannot guarantee detection of all transformed copies.
Native capture/session metadata is unavailable, so group checks use UEC image IDs,
Roboflow filename identities and exact/near-image content. Review the taxonomy and per-source
validation coverage before interpreting cross-dataset metrics. Training and test performance
are intentionally absent until the corresponding cells are explicitly enabled.
''')
nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python (ai-unified)', 'language': 'python', 'name': 'ai-unified'}, 'language_info': {'name': 'python', 'version': '3'}})
nbf.validate(nb)
(ROOT / 'notebooks').mkdir(exist_ok=True)
nbf.write(nb, ROOT / 'notebooks/EXP-001_UnifiedFood_4Datasets.ipynb')
