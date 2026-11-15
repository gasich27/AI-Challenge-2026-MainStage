"""Локальные признаки рёбер для artifacts."""

from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ['local_edge_p95', 'local_edge_p99',
            'local_edge_over_2', 'local_edge_over_4']
MAX_FILE_BYTES = 150_000_000
MAX_FACES = 12_000


def extract_one(item_id, data_dir):
    path = Path(data_dir) / f'{item_id}.npz'
    result = {'item_id': item_id, **dict.fromkeys(FEATURES, np.nan),
              'status': 'ok', 'sampled_faces': 0}

    if not path.is_file():
        result['status'] = 'missing_file'
        return result
    if path.stat().st_size > MAX_FILE_BYTES:
        result['status'] = 'oversize_file'
        return result

    try:
        with np.load(path, allow_pickle=False) as mesh:
            vertices = mesh['vertices']
            faces = mesh['faces']

        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            result['status'] = 'wrong_shape'
            return result
        if len(vertices) == 0 or len(faces) == 0:
            result['status'] = 'empty_mesh'
            return result

        if len(faces) > MAX_FACES:
            block = MAX_FACES // 4
            starts = np.linspace(0, len(faces) - block, 4, dtype=int)
            faces = np.concatenate([faces[start:start + block] for start in starts])

        valid = (faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1)
        faces = faces[valid]
        if len(faces) < 4:
            result['status'] = 'few_valid_faces'
            return result

        used = np.unique(faces)
        points = np.asarray(vertices[used], dtype=np.float64)
        if not np.isfinite(points).all():
            result['status'] = 'nonfinite_vertices'
            return result
        faces = np.searchsorted(used, faces)
        result['sampled_faces'] = len(faces)

        pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.unique(np.sort(pairs, axis=1), axis=0)
        length = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        positive = np.isfinite(length) & (length > 1e-12)
        edges, length = edges[positive], length[positive]
        if len(length) < 4:
            result['status'] = 'few_valid_edges'
            return result

        # Медиана длин всех рёбер, инцидентных каждой вершине.
        vertex_id = np.r_[edges[:, 0], edges[:, 1]]
        local = pd.Series(np.r_[length, length]).groupby(vertex_id).median()
        local_scale = (local.loc[edges[:, 0]].to_numpy() +
                       local.loc[edges[:, 1]].to_numpy()) / 2
        ratio = length / np.maximum(local_scale, 1e-12)

        result.update({
            'local_edge_p95': float(np.percentile(ratio, 95)),
            'local_edge_p99': float(np.percentile(ratio, 99)),
            'local_edge_over_2': float((ratio > 2).mean()),
            'local_edge_over_4': float((ratio > 4).mean()),
        })
    except (OSError, ValueError, KeyError, MemoryError):
        result['status'] = 'read_error'
    return result
