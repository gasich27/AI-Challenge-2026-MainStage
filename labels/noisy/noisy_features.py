"""Признаки изменения поверхности после слабого сглаживания."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from tqdm.auto import tqdm

from labels.lowpoly.geometry_features import MAX_FILE_BYTES, _face_sample


R1_FEATURES = [
    'smooth_r1_mean', 'smooth_r1_p50', 'smooth_r1_p90',
    'smooth_r1_p95', 'smooth_r1_p99', 'smooth_r1_local_p95',
    'smooth_r1_local_over_01',
]
MULTI_FEATURES = [
    'smooth_r3_mean', 'smooth_r3_p95', 'smooth_r3_p99',
    'smooth_r3_local_p95', 'smooth_r3_to_r1_p50',
    'smooth_r3_to_r1_p90', 'smooth_highfreq_local_p95',
]
FEATURES = R1_FEATURES + MULTI_FEATURES


def extract_one(item_id: str, data_dir: Path) -> dict:
    result = {'item_id': item_id, **dict.fromkeys(FEATURES, np.nan),
              'status': 'ok', 'sampled_faces': 0, 'valid_vertices': 0}
    path = data_dir / f'{item_id}.npz'
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        result['status'] = 'missing_or_large'
        return result

    try:
        with np.load(path, allow_pickle=False) as mesh:
            vertices, faces = mesh['vertices'], mesh['faces']
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            result['status'] = 'bad_shape'
            return result
        faces = _face_sample(faces.astype(np.int64))
        faces = faces[(faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1)]
        if len(faces) < 4:
            result['status'] = 'few_faces'
            return result
        triangle = vertices[faces].astype(np.float64)
        area2 = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0],
                                        triangle[:, 2] - triangle[:, 0]), axis=1)
        faces = faces[np.isfinite(triangle).all(axis=(1, 2)) & (area2 > 1e-12)]
        if len(faces) < 4:
            result['status'] = 'bad_faces'
            return result
        result['sampled_faces'] = len(faces)

        vertex_ids, mapped = np.unique(faces, return_inverse=True)
        points = vertices[vertex_ids].astype(np.float64)
        faces = mapped.reshape(-1, 3)
        pairs = np.sort(np.stack([faces[:, [0, 1]], faces[:, [1, 2]],
                                   faces[:, [2, 0]]], axis=1).reshape(-1, 2), axis=1)
        edges, counts = np.unique(pairs, axis=0, return_counts=True)
        u, v = edges.T
        size = len(points)
        adjacency = coo_matrix((np.ones(2 * len(edges)),
                                (np.r_[u, v], np.r_[v, u])), shape=(size, size)).tocsr()
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        boundary = np.zeros(size, dtype=bool)
        boundary[edges[counts != 2].ravel()] = True
        valid = (degree >= 3) & ~boundary
        if not valid.any():
            result['status'] = 'no_interior_vertices'
            return result
        result['valid_vertices'] = int(valid.sum())

        length = np.linalg.norm(points[u] - points[v], axis=1)
        local_scale = np.bincount(np.r_[u, v], weights=np.r_[length, length], minlength=size)
        local_scale /= np.maximum(degree, 1)
        diagonal = np.linalg.norm(np.ptp(points, axis=0))
        if diagonal <= 1e-12:
            result['status'] = 'zero_size'
            return result

        current = points.copy()
        residuals = []
        for step in range(3):
            average = (adjacency @ current) / np.maximum(degree[:, None], 1)
            current = .75 * current + .25 * average
            current[boundary] = points[boundary]
            if step in (0, 2):
                residuals.append(np.linalg.norm(points - current, axis=1)[valid])
        r1, r3 = residuals
        scale = np.maximum(local_scale[valid], 1e-12)
        a, b = r1 / diagonal, r3 / diagonal
        local1, local3 = np.minimum(r1 / scale, 100), np.minimum(r3 / scale, 100)
        ratio = np.minimum(r3 / np.maximum(r1, 1e-12), 100)

        result.update({
            'smooth_r1_mean': float(a.mean()), 'smooth_r1_p50': float(np.percentile(a, 50)),
            'smooth_r1_p90': float(np.percentile(a, 90)),
            'smooth_r1_p95': float(np.percentile(a, 95)),
            'smooth_r1_p99': float(np.percentile(a, 99)),
            'smooth_r1_local_p95': float(np.percentile(local1, 95)),
            'smooth_r1_local_over_01': float((local1 > .1).mean()),
            'smooth_r3_mean': float(b.mean()),
            'smooth_r3_p95': float(np.percentile(b, 95)),
            'smooth_r3_p99': float(np.percentile(b, 99)),
            'smooth_r3_local_p95': float(np.percentile(local3, 95)),
            'smooth_r3_to_r1_p50': float(np.percentile(ratio, 50)),
            'smooth_r3_to_r1_p90': float(np.percentile(ratio, 90)),
            'smooth_highfreq_local_p95': float(np.percentile(np.maximum(r1 - r3 / 3, 0) / scale, 95)),
        })
        if not np.isfinite([result[name] for name in FEATURES]).all():
            result['status'] = 'invalid_features'
            result.update(dict.fromkeys(FEATURES, np.nan))
    except (OSError, ValueError, KeyError, MemoryError):
        result['status'] = 'read_error'
    return result


def extract_all(item_ids, data_dir: Path, output_path: Path) -> pd.DataFrame:
    rows = pd.read_csv(output_path).to_dict('records') if output_path.is_file() else []
    done = {row['item_id'] for row in rows}
    missing = [item_id for item_id in item_ids if item_id not in done]
    with ThreadPoolExecutor(max_workers=3) as pool:
        extracted = pool.map(lambda item_id: extract_one(item_id, data_dir), missing)
        for count, row in enumerate(tqdm(extracted, total=len(missing), desc='Smoothing'), 1):
            rows.append(row)
            if count % 100 == 0:
                pd.DataFrame(rows).to_csv(output_path, index=False)
    table = pd.DataFrame(rows).set_index('item_id')
    table.to_csv(output_path)
    return table.loc[item_ids]
