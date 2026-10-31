"""Дополнительные признаки меша для эксперимента lowpoly.

Для больших мешей берём четыре непрерывных участка списка граней. Это
сохраняет настоящие пары соседних граней внутри участков, но признаки
соседства становятся приближёнными. Объёмы выборок записываются в CSV.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


MAX_FILE_BYTES = 150_000_000
MAX_FACES = 12_000
MAX_PCA_VERTICES = 15_000
PCA_QUERIES = 128

NEW_FEATURES = [
    'edge_min_relative', 'edge_p10_relative', 'edge_p50_relative',
    'edge_p90_relative', 'edge_p95_relative', 'edge_long_share',
    'area_p10_relative', 'area_p50_relative', 'area_p90_relative',
    'area_p95_relative', 'area_max_to_median', 'area_tiny_share',
    'triangle_aspect_p50', 'triangle_aspect_p90', 'triangle_thin_share',
    'triangle_min_angle_p10', 'triangle_min_angle_p50',
    'triangle_max_angle_p90', 'triangle_near_degenerate_share',
    'dihedral_mean', 'dihedral_std', 'dihedral_p50', 'dihedral_p75',
    'dihedral_p90', 'dihedral_p95', 'dihedral_max',
    'dihedral_over_10_share', 'dihedral_over_30_share',
    'dihedral_over_60_share', 'dihedral_edge_weighted_mean',
    'dihedral_edge_weighted_p90',
    'curvature_mean', 'curvature_p90', 'curvature_p95',
    'normal_variation_p50', 'normal_variation_p90',
    'normal_variation_p95', 'normal_variation_std',
    'pca_curvature_mean', 'pca_curvature_p90', 'pca_curvature_p95',
    'boundary_edge_share', 'nonmanifold_edge_share',
    'faces_per_bbox_volume',
]


def _percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else np.nan


def _weighted_percentile(values, weights, q):
    if not len(values) or weights.sum() <= 0:
        return np.nan
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, q * cumulative[-1])]])


def _face_sample(faces, limit=MAX_FACES):
    if len(faces) <= limit:
        return faces
    block = limit // 4
    starts = np.linspace(0, len(faces) - block, 4, dtype=np.int64)
    return np.concatenate([faces[start:start + block] for start in starts])


def _local_pca(vertices, rng):
    finite = vertices[np.isfinite(vertices).all(axis=1)]
    if len(finite) < 24:
        return np.array([])
    if len(finite) > MAX_PCA_VERTICES:
        finite = finite[rng.choice(len(finite), MAX_PCA_VERTICES, replace=False)]
    queries = finite[rng.choice(len(finite), min(PCA_QUERIES, len(finite)), replace=False)]
    tree = cKDTree(finite)
    _, neighbours = tree.query(queries, k=min(24, len(finite)), workers=1)
    clouds = finite[neighbours]
    centered = clouds - clouds.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered) / clouds.shape[1]
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    return eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-15)


def extract_one(item_id: str, train_dir: Path) -> dict:
    path = train_dir / f'{item_id}.npz'
    result = {'item_id': item_id, **dict.fromkeys(NEW_FEATURES, np.nan),
              'status': 'ok', 'original_faces': np.nan, 'sampled_faces': 0,
              'invalid_faces': 0, 'degenerate_faces': 0,
              'duplicate_vertex_share': np.nan, 'interior_edges': 0}
    if not path.is_file():
        result['status'] = 'missing_file'
        return result
    if path.stat().st_size > MAX_FILE_BYTES:
        result['status'] = 'oversize_file'
        return result

    seed = int.from_bytes(hashlib.blake2b(item_id.encode(), digest_size=8).digest(), 'little')
    rng = np.random.default_rng(seed)
    try:
        with np.load(path, allow_pickle=False) as mesh:
            vertices = np.asarray(mesh['vertices'], dtype=np.float64)
            faces = np.asarray(mesh['faces'], dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            result['status'] = 'wrong_shape'
            return result
        if not len(vertices) or not len(faces):
            result['status'] = 'empty_mesh'
            return result
        result['original_faces'] = len(faces)
        sampled = _face_sample(faces)
        valid_index = ((sampled >= 0) & (sampled < len(vertices))).all(axis=1)
        result['invalid_faces'] = int((~valid_index).sum())
        sampled = sampled[valid_index]
        finite_vertices = np.isfinite(vertices).all(axis=1)
        sampled = sampled[finite_vertices[sampled].all(axis=1)]
        result['sampled_faces'] = len(sampled)
        if not len(sampled):
            result['status'] = 'no_valid_faces'
            return result

        finite = vertices[finite_vertices]
        if len(finite) > MAX_PCA_VERTICES:
            finite = finite[rng.choice(len(finite), MAX_PCA_VERTICES, replace=False)]
        result['duplicate_vertex_share'] = 1 - len(np.unique(finite, axis=0)) / len(finite)
        # Отношение собственных значений локальной PCA — приближение кривизны.
        pca = _local_pca(vertices, rng)
        if len(pca):
            result['pca_curvature_mean'] = float(pca.mean())
            result['pca_curvature_p90'] = _percentile(pca, 90)
            result['pca_curvature_p95'] = _percentile(pca, 95)

        tri = vertices[sampled]
        a = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
        b = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
        c = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        twice_area = np.linalg.norm(cross, axis=1)
        area = twice_area / 2
        sizes = np.ptp(vertices[np.isfinite(vertices).all(axis=1)], axis=0)
        diagonal = float(np.linalg.norm(sizes))
        scale = max(diagonal, 1e-12)
        eps_area = max(diagonal ** 2 * 1e-12, 1e-20)
        good = (area > eps_area) & (np.minimum.reduce([a, b, c]) > scale * 1e-12)
        result['degenerate_faces'] = int((~good).sum())
        if not good.any():
            result['status'] = 'zero_area'
            return result

        area_good = area[good]
        lengths = np.concatenate([a[good], b[good], c[good]])
        median_edge = np.median(lengths)
        for name, value in [('edge_min_relative', lengths.min()),
                            ('edge_p10_relative', _percentile(lengths, 10)),
                            ('edge_p50_relative', median_edge),
                            ('edge_p90_relative', _percentile(lengths, 90)),
                            ('edge_p95_relative', _percentile(lengths, 95))]:
            result[name] = float(value / scale)
        result['edge_long_share'] = float(np.mean(lengths > 3 * median_edge))
        median_area = np.median(area_good)
        for q in (10, 50, 90, 95):
            result[f'area_p{q}_relative'] = float(_percentile(area_good, q) / scale ** 2)
        result['area_max_to_median'] = float(area_good.max() / max(median_area, 1e-20))
        result['area_tiny_share'] = float(np.mean(area_good < 0.01 * median_area))
        result['triangle_near_degenerate_share'] = float(np.mean(area <= eps_area))

        sides = np.stack([a[good], b[good], c[good]], axis=1)
        aspect = sides.max(axis=1) / np.maximum(sides.min(axis=1), 1e-12)
        result['triangle_aspect_p50'] = _percentile(aspect, 50)
        result['triangle_aspect_p90'] = _percentile(aspect, 90)
        result['triangle_thin_share'] = float(np.mean(aspect > 5))
        angle_a = np.degrees(np.arccos(np.clip((b[good] ** 2 + c[good] ** 2 - a[good] ** 2)
                                               / np.maximum(2 * b[good] * c[good], 1e-20), -1, 1)))
        angle_b = np.degrees(np.arccos(np.clip((a[good] ** 2 + c[good] ** 2 - b[good] ** 2)
                                               / np.maximum(2 * a[good] * c[good], 1e-20), -1, 1)))
        angle_c = 180 - angle_a - angle_b
        min_angle = np.minimum.reduce([angle_a, angle_b, angle_c])
        max_angle = np.maximum.reduce([angle_a, angle_b, angle_c])
        result['triangle_min_angle_p10'] = _percentile(min_angle, 10)
        result['triangle_min_angle_p50'] = _percentile(min_angle, 50)
        result['triangle_max_angle_p90'] = _percentile(max_angle, 90)

        # Каждое ребро связываем с реальными индексами его граней.
        f = sampled[good]
        edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        face_ids = np.tile(np.arange(len(f)), 3)
        edges.sort(axis=1)
        nonzero = edges[:, 0] != edges[:, 1]
        edges, face_ids = edges[nonzero], face_ids[nonzero]
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges, face_ids = edges[order], face_ids[order]
        starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1]
        counts = np.diff(np.r_[starts, len(edges)])
        result['boundary_edge_share'] = float(np.mean(counts == 1))
        result['nonmanifold_edge_share'] = float(np.mean(counts > 2))
        interior = starts[counts == 2]
        result['interior_edges'] = len(interior)
        if len(interior):
            n = cross[good] / twice_area[good, None]
            face_a, face_b = face_ids[interior], face_ids[interior + 1]
            cos = np.abs(np.einsum('ij,ij->i', n[face_a], n[face_b]))
            angles = np.degrees(np.arccos(np.clip(cos, 0, 1)))
            edge_pairs = edges[interior]
            edge_lengths = np.linalg.norm(vertices[edge_pairs[:, 0]] - vertices[edge_pairs[:, 1]], axis=1)
            result['dihedral_mean'] = float(angles.mean())
            result['dihedral_std'] = float(angles.std())
            result['dihedral_max'] = float(angles.max())
            for q in (50, 75, 90, 95):
                result[f'dihedral_p{q}'] = _percentile(angles, q)
            for threshold in (10, 30, 60):
                result[f'dihedral_over_{threshold}_share'] = float(np.mean(angles > threshold))
            result['dihedral_edge_weighted_mean'] = float(np.average(angles, weights=np.maximum(edge_lengths, 1e-12)))
            result['dihedral_edge_weighted_p90'] = _weighted_percentile(angles, edge_lengths, .9)
            curvature = angles / np.maximum(edge_lengths / scale, 1e-9)
            result['curvature_mean'] = float(curvature.mean())
            result['curvature_p90'] = _percentile(curvature, 90)
            result['curvature_p95'] = _percentile(curvature, 95)
            per_face = np.bincount(np.r_[face_a, face_b], weights=np.r_[angles, angles], minlength=len(f))
            degree = np.bincount(np.r_[face_a, face_b], minlength=len(f))
            variation = per_face[degree > 0] / degree[degree > 0]
            result['normal_variation_p50'] = _percentile(variation, 50)
            result['normal_variation_p90'] = _percentile(variation, 90)
            result['normal_variation_p95'] = _percentile(variation, 95)
            result['normal_variation_std'] = float(variation.std())

        volume = float(np.prod(sizes))
        if volume > diagonal ** 3 * 1e-9:
            result['faces_per_bbox_volume'] = float(len(faces) * diagonal ** 3 / volume)
        if len(faces) > MAX_FACES:
            result['status'] = 'sampled'
    except (OSError, ValueError, KeyError, IndexError, MemoryError) as error:
        result['status'] = type(error).__name__

    for name in NEW_FEATURES:
        if not np.isfinite(result[name]):
            result[name] = np.nan
    return result
