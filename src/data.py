import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def split_indices(labels):
    if labels.item_id.isna().any() or not labels.item_id.is_unique:
        raise ValueError('Пустые или повторяющиеся item_id')
    targets = labels.drop(columns='item_id')
    if not targets.isin([0, 1]).all().all():
        raise ValueError('В разметке должны быть только 0 и 1')
    outer = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.15, random_state=42)
    train_val, test = next(outer.split(labels, targets))
    inner = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.1765, random_state=43)
    train, val = next(inner.split(labels.iloc[train_val], targets.iloc[train_val]))
    return train_val[train], train_val[val], test


def split_data(labels):
    return tuple(labels.iloc[i].reset_index(drop=True) for i in split_indices(labels))


def align_features(table, item_ids):

    table = table.set_index('item_id') if 'item_id' in table else table
    ids = pd.Index(item_ids)
    if not table.index.is_unique or not ids.is_unique or ids.isna().any():
        raise ValueError('Повторяющиеся или пустые ID')
    missing = ids.difference(table.index)
    if len(missing):
        raise ValueError(f'Нет признаков для {len(missing)} объектов')
    return table.loc[ids]


def load_embeddings(path, manifest_path, done_path, item_ids, expected, labels_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf8'))
    if manifest.get('item_ids') != list(item_ids):
        raise ValueError('Порядок ID в embeddings не совпал')
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Модель или обработка изображений в кэше другая')
    digest = hashlib.sha256(Path(labels_path).read_bytes()).hexdigest()
    if manifest.get('labels_sha256') != digest:
        raise ValueError('Кэш рассчитан для другой разметки')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    done = np.load(done_path, allow_pickle=False)
    if 'dtype' in manifest and str(values.dtype) != manifest['dtype']:
        raise ValueError('Тип массива отличается от manifest')
    if values.shape != (len(item_ids), 6, 384) or done.shape != (len(item_ids),) or not done.all():
        raise ValueError('Размер кэша не совпал или расчёт не закончен')
    for start in range(0, len(values), 256):
        if not np.isfinite(values[start:start + 256]).all():
            raise ValueError('В embeddings есть NaN или бесконечность')
    return values
