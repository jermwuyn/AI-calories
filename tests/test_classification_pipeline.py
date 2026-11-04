import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from classification.metrics import Metrics

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('classification_cli', ROOT / 'scripts' / 'classification.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(4, 5), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(5, 3))

    def forward(self, x):
        return self.classifier(self.features(x))


class PipelineTests(unittest.TestCase):
    def test_metrics_known_confusion_matrix(self):
        metrics = Metrics(3)
        metrics.update(torch.tensor([[3., 2., 1.], [1., 3., 2.], [3., 2., 1.]]),
                       torch.tensor([0, 1, 2]), torch.tensor(1.))
        result = metrics.result([{'source_name': str(i)} for i in range(3)])
        self.assertAlmostEqual(result['top1_accuracy'], 2 / 3)
        self.assertEqual(result['top3_accuracy'], 1.0)
        self.assertAlmostEqual(result['macro_f1'], 5 / 9)
        self.assertEqual(result['confusion_matrix'], [[1, 0, 0], [0, 1, 0], [1, 0, 0]])

    def test_resume_matches_uninterrupted_two_stages(self):
        cfg = {'seed': 42, 'amp': False, 'dataset': 'synthetic', 'architecture': 'tiny',
               'pretrained': False, 'batch_size': 3, 'workers': 0, 'head_epochs': 1,
               'finetune_epochs': 1, 'head_lr': 0.01, 'finetune_lr': 0.001,
               'weight_decay': 0.0, 'gradient_clip': 1.0, 'patience': 5}
        mapping = {'head': 'synthetic', 'classes': [{'source_name': str(i)} for i in range(3)]}
        data = TensorDataset(torch.arange(24, dtype=torch.float32).reshape(6, 4) / 24,
                             torch.tensor([0, 1, 2, 0, 1, 2]))
        def load(_cfg, _splits):
            return ROOT, mapping, {split: data for split in _splits}
        def build(*args):
            return TinyModel()
        real_epoch = cli.run_epoch
        calls = [0]
        def interrupt(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError('simulated interruption')
            return real_epoch(*args, **kwargs)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, patch.object(cli, 'load_datasets', load), \
                patch.object(cli, 'build_model', build), patch('torch.cuda.is_available', return_value=False):
            baseline, resumed = Path(tmp) / 'baseline', Path(tmp) / 'resumed'
            cli.train(cfg, baseline, None)
            with patch.object(cli, 'run_epoch', interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    cli.train(cfg, resumed, None)
            cli.train(cfg, resumed, resumed / 'last.pth')
            a = torch.load(baseline / 'last.pth', weights_only=False, map_location='cpu')
            b = torch.load(resumed / 'last.pth', weights_only=False, map_location='cpu')
            self.assertEqual(a['epoch'], b['epoch'])
            self.assertEqual(a['history'], b['history'])
            self.assertTrue(all(torch.equal(a['model'][k], b['model'][k]) for k in a['model']))
            self.assertEqual(a['scheduler'], b['scheduler'])
            self.assertTrue((resumed / 'best.pth').is_file())
            self.assertTrue((resumed / 'metrics.json').is_file())
            result_path = resumed / 'eval.json'
            cli.evaluate(resumed / 'best.pth', 'test', result_path)
            result = json.loads(result_path.read_text(encoding='utf-8'))
            self.assertEqual(result['split'], 'test')
            self.assertEqual(len(result['confusion_matrix']), 3)
            self.assertGreater(result['inference_latency_ms_batch1'], 0)


if __name__ == '__main__':
    unittest.main()
