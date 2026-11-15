import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from src.data import align_features, load_embeddings, split_indices
from src.evaluation import binary_metrics, best_threshold
from labels.artifacts.geometry_features import extract_one as local_edges
from labels.artifacts.component_features import extract_one as components
from labels.lowpoly.geometry_features import extract_one as geometry
from labels.noisy.noisy_features import extract_one as smoothing


class CoreTests(unittest.TestCase):
    def test_metrics(self):
        m = binary_metrics([0, 0, 1, 1], [.1, .8, .7, .2])
        self.assertEqual([m[k] for k in ['TP', 'TN', 'FP', 'FN']], [1, 1, 1, 1])
        self.assertEqual(m['F1'], .5)
        self.assertEqual(binary_metrics([1, 1], [0, 0])['F1'], 0)
        self.assertTrue(np.isnan(binary_metrics([0, 0], [.1, .2])['AP']))
        self.assertTrue(np.isnan(binary_metrics([1, 1], [.1, .2])['ROC_AUC']))
        self.assertEqual(best_threshold([0, 1], [.1, .9], [.2, .5]), .2)
        with self.assertRaises(ValueError):
            binary_metrics([0], [np.nan])

    def test_alignment(self):
        table = pd.DataFrame({'item_id': ['b', 'a'], 'x': [2, 1]})
        self.assertEqual(align_features(table, ['a', 'b']).x.tolist(), [1, 2])
        with self.assertRaises(ValueError):
            align_features(table, ['c'])
        with self.assertRaises(ValueError):
            align_features(table, ['a', 'a'])

    def test_split(self):
        path = Path(__file__).resolve().parents[1] / 'data/train.csv'
        if not path.is_file():
            self.skipTest('Нет локального data/train.csv')
        labels = pd.read_csv(path)
        a, b, c = split_indices(labels)
        self.assertEqual(tuple(map(len, [a, b, c])), (6259, 1361, 1344))
        self.assertEqual(len(set(a) | set(b) | set(c)), len(labels))
        self.assertFalse(set(a) & set(b) or set(a) & set(c) or set(b) & set(c))

    def test_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d / 'labels.csv').write_text('item_id,label\na,1\n')
            digest = hashlib.sha256((d / 'labels.csv').read_bytes()).hexdigest()
            (d / 'meta.json').write_text(json.dumps({'item_ids': ['a'], 'model': 'test', 'labels_sha256': digest}))
            np.save(d / 'values.npy', np.zeros((1, 6, 384)))
            np.save(d / 'done.npy', [True])
            args = [d/'values.npy', d/'meta.json', d/'done.npy', ['a'], {'model': 'test'}, d/'labels.csv']
            self.assertEqual(load_embeddings(*args).shape, (1, 6, 384))
            args[3] = ['other']
            with self.assertRaises(ValueError):
                load_embeddings(*args)

    def test_geometry_cases(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        cases = {
            'tetra': (vertices, faces),
            'empty': (np.empty((0, 3)), np.empty((0, 3), dtype=int)),
            'indices': (vertices, np.array([[0, 1, 99]])),
            'degenerate': (vertices, np.zeros((4, 3), dtype=int)),
            'flat': (vertices * [1, 1, 0], faces),
            'nan': (vertices * np.nan, faces),
            'disconnected': (np.r_[vertices, vertices + 3], np.r_[faces, faces + 4]),
            'nonmanifold': (vertices, np.r_[faces, faces[:1]]),
            'float_indices': (vertices, faces.astype(float) + .1),
            'wrong_shape': (vertices, faces[:, :2]),
            'duplicates': (np.r_[vertices, vertices], faces),
        }
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            for name, (v, f) in cases.items():
                np.savez(d / f'{name}.npz', vertices=v, faces=f)
                for extract in [local_edges, components, geometry, smoothing]:
                    row = extract(name, d)
                    numbers = [v for v in row.values() if isinstance(v, (int, float))]
                    self.assertFalse(np.isinf(numbers).any(), (name, extract.__module__))
            self.assertEqual(components('disconnected', d)['component_count'], 2)
            self.assertEqual(components('tetra', d)['component_count'], 1)


if __name__ == '__main__':
    unittest.main()
