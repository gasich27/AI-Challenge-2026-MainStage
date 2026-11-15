"""Признаки связных частей меша по индексам вершин."""

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


FEATURES = ['component_count', 'largest_component_share',
            'small_component_share', 'far_small_components']
MAX_FILE_BYTES = 150_000_000
MAX_FACES = 500_000
MAX_VERTICES = 750_000


def extract_one(item_id, data_dir):
    path = Path(data_dir) / f'{item_id}.npz'
    result = {'item_id': item_id, **dict.fromkeys(FEATURES, np.nan),
              'status': 'ok'}
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
        if len(faces) > MAX_FACES or len(vertices) > MAX_VERTICES:
            result['status'] = 'over_limit'
            return result

        valid = (faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1)
        faces = faces[valid]
        distinct = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) &
                    (faces[:, 2] != faces[:, 0]))
        faces = faces[distinct]
        if len(faces) == 0:
            result['status'] = 'no_valid_faces'
            return result
        if not np.isfinite(vertices).all():
            result['status'] = 'nonfinite_vertices'
            return result

        pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8),
                            (pairs[:, 0], pairs[:, 1])),
                           shape=(len(vertices), len(vertices))).tocsr()
        n_components, labels = connected_components(graph, directed=False)
        face_labels = labels[faces[:, 0]]
        counts = np.bincount(face_labels, minlength=n_components)
        largest = int(counts.argmax())
        present = counts > 0
        small = present & (counts < max(10, int(.001 * len(faces))))
        small[largest] = False

        diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        face_centers = vertices[faces].mean(axis=1)
        centers = np.column_stack([
            np.bincount(face_labels, weights=face_centers[:, j],
                        minlength=n_components) / np.maximum(counts, 1)
            for j in range(3)
        ])
        distance = np.linalg.norm(centers - centers[largest], axis=1)
        result.update({
            'component_count': int(present.sum()),
            'largest_component_share': float(counts[largest] / len(faces)),
            'small_component_share': float(counts[small].sum() / len(faces)),
            'far_small_components': int((small & (distance > .1 * max(diagonal, 1e-12))).sum()),
        })
    except (OSError, ValueError, KeyError, MemoryError):
        result['status'] = 'read_error'
    return result
