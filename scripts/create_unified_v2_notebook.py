"""Create the V2 EXP-001 notebook without changing the original V1 notebook."""
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'notebooks/EXP-001_UnifiedFood_4Datasets.ipynb'
destination = ROOT/'notebooks/EXP-001_UnifiedFood_4Datasets_V2.ipynb'
nb = nbformat.read(source, as_version=4)
nb.cells[0].source = nb.cells[0].source.replace('4 original datasets', '4 original datasets — taxonomy V2') + '\n\nThis notebook uses the reviewed 310-class V2 mapping. Full training and test evaluation remain disabled by default.'
nb.cells[1].source = nb.cells[1].source.replace('configs/classification-unified-exp001.yaml','configs/classification-unified-v2.yaml').replace('runs/exp-001-unified4','runs/exp-001-unified4-v2')
nb.cells[5].source = '''report = json.loads((DATA / 'report.json').read_text(encoding='utf-8'))
inventory = json.loads((DATA / 'source_inventory.json').read_text(encoding='utf-8'))
display(pd.DataFrame(inventory['sources']).T.fillna(0))
display(pd.DataFrame(report['samples_by_source']).T.fillna(0).astype(int))
print(report['image_audit'])
display(pd.Series(report['exclusions_inherited_from_v1'], name='excluded in V1'))
'''
nb.cells[6].source = '## 4. Unified class mapping\nSeven evidence-backed merges are applied; unresolved pairs remain separate. Stable canonical IDs are distinct from contiguous model indices.'
nb.cells[7].source = '''root, mapping, datasets = load_unified(ROOT, CFG, ('train', 'val', 'test'))
assert len(mapping['classes']) == 310
display(pd.DataFrame(mapping['provenance']))
print('Canonical classes:', len(mapping['classes']))
print('V2 review decisions:', report['decisions'])
display(pd.read_csv(DATA / 'taxonomy_decisions.csv').head(21))
'''
nb.cells[8].source = '## 5. Class distribution\nV1 ID 307 remains `food-v2-307` without train samples. V1 ID 310 maps to `food-v2-018`, which has 477 train samples from 30VNFoods. VinaFood21 has no native validation split.'
nb.cells[17].source = nb.cells[17].source.replace('runs/exp-001-unified4/tensorboard','runs/exp-001-unified4-v2/tensorboard')
nb.cells[26].source = '''## 14. Conclusions
V2 retains the same four-source samples and splits as V1 and applies reviewed
taxonomy merges. No Nutrition5k data is included. Near-duplicate detection is
heuristic, source capture metadata is limited, and one V2 class has no train
examples. VinaFood21 has no native validation split. Training, test metrics,
and inference appear only after the corresponding cells are explicitly enabled.
'''
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
nbformat.validate(nb)
nbformat.write(nb, destination)
print(destination)
