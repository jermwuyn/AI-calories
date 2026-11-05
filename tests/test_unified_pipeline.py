import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from PIL import Image
from classification.unified import ArchiveReader, open_sample, UnifiedDataset
from scripts.prepare_unified import HammingIndex, Groups, normalized, yolo_region


class UnifiedTests(unittest.TestCase):
    def test_yolo_polygon_and_invalid_geometry(self):
        self.assertEqual(yolo_region([0, .1, .1, .9, .1, .9, .9, .1, .9], 100, 100), [10., 10., 90., 90.])
        for values in ([0, .1, .1, .2, .2, .3, .3], [0, -1, 0, 1, 0, 1, 1], [0, float('nan'), 0, 1, 1]):
            with self.assertRaises(ValueError): yolo_region(values, 100, 100)

    def test_near_search_includes_all_identical_hashes(self):
        tree = HammingIndex()
        self.assertEqual(tree.query_add(0, 0), [])
        self.assertEqual(tree.query_add(0, 1), [0])
        self.assertEqual(set(tree.query_add(1, 2)), {0, 1})
        self.assertEqual(tree.query_add(2**64-1, 3), [])

    def test_hamming_index_matches_brute_force(self):
        import random
        rng = random.Random(19)
        values = [rng.getrandbits(64) for _ in range(200)]
        values += [value ^ sum(1 << bit for bit in rng.sample(range(64), 4)) for value in values[:50]]
        index = HammingIndex()
        for i, value in enumerate(values):
            expected = [j for j in range(i) if (value ^ values[j]).bit_count() <= 4]
            self.assertEqual(index.query_add(value, i), expected)

    def test_transitive_group(self):
        g = Groups(4)
        g.union(0, 1); g.union(1, 2)
        self.assertEqual(g.find(0), g.find(2))
        self.assertNotEqual(g.find(0), g.find(3))

    def test_name_variants_not_collapsed(self):
        self.assertEqual(normalized('banh-can'), normalized('Banh can'))
        self.assertNotEqual(normalized('Beef Pho'), normalized('Pho'))
        self.assertNotEqual(normalized('Grilled Pork with Vermicelli'), normalized('Grilled Pork Vermicelli'))

    def test_archive_multiple_crops_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'source.zip'
            data = np.zeros((40, 80, 3), dtype=np.uint8)
            data[:, :40, 0] = 255
            buf = io.BytesIO(); Image.fromarray(data).save(buf, format='PNG')
            with zipfile.ZipFile(archive, 'w') as z: z.writestr('image.png', buf.getvalue())
            row = {'archive': str(archive), 'member': 'image.png', 'sample_id': 'one', 'bbox': [0, 0, 40, 40]}
            reader = ArchiveReader()
            a = open_sample(row, reader)
            b = open_sample(dict(row, bbox=[40, 0, 80, 40]), reader)
            self.assertEqual(a.size, (40, 40))
            self.assertNotEqual(a.getpixel((0, 0)), b.getpixel((0, 0)))
            for box in ([-1, 0, 40, 40], [0, 0, 81, 40], [0, 0, 0, 40], [0, 0, 8, 8]):
                with self.assertRaises(ValueError): open_sample(dict(row, bbox=box), reader)
            for handle in reader.handles.values(): handle.close()

    def test_manifest_rejects_relocated_test_and_wrong_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping = {'classes': [{'index': 0}], 'provenance': [{'source': 'a', 'source_class_id': '1', 'canonical_id': 0}]}
            row = {'group_id': 'g', 'sample_id': 's', 'split': 'train', 'source_split': 'test', 'source': 'a', 'source_class_id': '1', 'canonical_id': 0}
            (root / 'train.jsonl').write_text(json.dumps(row))
            with self.assertRaises(ValueError): UnifiedDataset(root, mapping, 'train')
            row.update(source_split='train', canonical_id=2)
            (root / 'train.jsonl').write_text(json.dumps(row))
            with self.assertRaises(ValueError): UnifiedDataset(root, mapping, 'train')


if __name__ == '__main__': unittest.main()
