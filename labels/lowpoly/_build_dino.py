from pathlib import Path
from textwrap import dedent

import nbformat as nb


def md(text):
    return nb.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nb.v4.new_code_cell(dedent(text).strip())


cells = [
    md('''
    # lowpoly: DINOv2 и геометрия

    Используем готовые embeddings шести ракурсов из эксперимента `simple`. DINOv2 здесь не запускается: обучается только CatBoost. Сравниваем изображения, геометрию и их объединение на той же validation. Internal test не трогаем.
    '''),
    md('## 1. Данные и сохранённые embeddings'),
    code('''
    %pip install -q catboost iterative-stratification

    from pathlib import Path
    import hashlib
    import io
    import json
    import zipfile
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    from catboost import CatBoostClassifier
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)

    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        pass

    ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
                 if (p / 'data/train.csv').is_file()), None)
    if ROOT is None:
        ROOT = Path('/content/drive/MyDrive/Colab Notebooks/colab_upload/AI challenge main')
    if not (ROOT / 'data/train.csv').is_file():
        ROOT = Path(input('Путь к папке проекта с data/train.csv: ').strip())
    assert (ROOT / 'data/train.csv').is_file(), ROOT

    package = ROOT / 'lowpoly_dinov2_assets.zip'
    if package.is_file():
        with zipfile.ZipFile(package) as archive:
            archive.extractall(ROOT)

    labels = pd.read_csv(ROOT / 'data/train.csv')
    targets = labels.columns.drop('item_id').tolist()
    split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.15, random_state=42)
    train_val_index, test_index = next(split.split(labels, labels[targets]))
    train_val = labels.iloc[train_val_index].reset_index(drop=True)
    split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.1765, random_state=43)
    train_index, val_index = next(split.split(train_val, train_val[targets]))
    train_df = train_val.iloc[train_index].reset_index(drop=True)
    val_df = train_val.iloc[val_index].reset_index(drop=True)
    assert (len(train_df), len(val_df), len(test_index)) == (6259, 1361, 1344)
    assert set(train_df.item_id).isdisjoint(val_df.item_id)
    objects = pd.concat([train_df, val_df], ignore_index=True)

    cache = ROOT / 'labels/simple/models/dinov2'
    for file in ('embeddings.npy', 'embeddings_done.npy', 'embeddings_manifest.json'):
        assert (cache / file).is_file(), f'Нет {file}: сначала завершите simple_dinov2.ipynb в Colab'
    manifest = json.loads((cache / 'embeddings_manifest.json').read_text(encoding='utf-8'))
    assert manifest['model'] == 'dinov2_vits14'
    assert manifest['item_ids'] == objects.item_id.tolist(), 'Порядок объектов не совпал'
    assert manifest['labels_sha256'] == hashlib.sha256((ROOT / 'data/train.csv').read_bytes()).hexdigest()
    assert np.load(cache / 'embeddings_done.npy').all(), 'Кэш не завершён'
    embeddings = np.load(cache / 'embeddings.npy', mmap_mode='r')
    assert embeddings.shape == (7620, 6, 384)
    print('Train:', len(train_df), 'Validation:', len(val_df), '| embeddings:', embeddings.shape)
    '''),
    md('## 2. Геометрия из предыдущего эксперимента'),
    code('''
    lowpoly = ROOT / 'labels/lowpoly'
    artifacts = lowpoly / 'artifacts'
    out = lowpoly / 'models/dinov2'
    out.mkdir(parents=True, exist_ok=True)
    needed = ['experiment_config.json', 'geometry_features.csv', 'validation_probabilities.csv']
    for name in needed:
        assert (artifacts / name).is_file(), f'Нет {name}: загрузите lowpoly_dinov2_assets.zip'

    config = json.loads((artifacts / 'experiment_config.json').read_text(encoding='utf-8'))
    columns = config['experiments']['A_no_curvature']['columns']
    base = pd.read_csv(ROOT / 'archive/artifacts/train_mesh_features_relative.csv').set_index('item_id')
    extra = pd.read_csv(artifacts / 'geometry_features.csv').set_index('item_id')
    assert base.index.is_unique and extra.index.is_unique
    geometry = base.join(extra[[name for name in columns if name not in base]], how='inner')
    geometry = geometry.loc[objects.item_id, columns].to_numpy(dtype=np.float32)
    assert geometry.shape == (7620, 78) and not np.isinf(geometry).any()

    old = pd.read_csv(artifacts / 'validation_probabilities.csv').set_index('item_id')
    old = old.loc[val_df.item_id]
    assert np.array_equal(old.lowpoly.to_numpy(), val_df.lowpoly.to_numpy())
    old_probability = old.A_no_curvature.to_numpy()
    old_threshold = config['experiments']['A_no_curvature']['threshold']
    print('Геометрия:', geometry.shape, '| старый F1:', round(f1_score(val_df.lowpoly, old_probability >= old_threshold), 4))
    '''),
    md('## 3. DINOv2 → CatBoost и объединение с геометрией'),
    code('''
    views = np.asarray(embeddings)
    mean = views.mean(axis=1)
    mean_max = np.concatenate([mean, views.max(axis=1)], axis=1)
    features = {
        'D1_mean': mean,
        'D2_mean_max': mean_max,
        'D3_mean_max_geometry': np.concatenate([mean_max, geometry], axis=1),
    }
    train_rows = np.arange(len(train_df))
    val_rows = np.arange(len(train_df), len(objects))
    y_train = train_df.lowpoly.to_numpy()
    y_val = val_df.lowpoly.to_numpy()
    parameters = json.loads((ROOT / 'archive/artifacts/catboost_features_config.json').read_text(encoding='utf-8'))['parameters']
    grid = np.arange(.05, .96, .01)

    def evaluate(probability, threshold):
        prediction = probability >= threshold
        tn, fp, fn, tp = confusion_matrix(y_val, prediction, labels=[0, 1]).ravel()
        return {'Precision': precision_score(y_val, prediction, zero_division=0),
                'Recall': recall_score(y_val, prediction, zero_division=0),
                'F1': f1_score(y_val, prediction, zero_division=0),
                'AP': average_precision_score(y_val, probability),
                'FP': int(fp), 'FN': int(fn), 'Threshold': float(threshold)}

    results = [{'Experiment': 'Geometry previous', **evaluate(old_probability, old_threshold)}]
    probabilities = val_df[['item_id', 'lowpoly']].copy()
    probabilities['Geometry_previous'] = old_probability
    for name, X in features.items():
        model = CatBoostClassifier(**parameters)
        model.fit(X[train_rows], y_train, eval_set=(X[val_rows], y_val),
                  early_stopping_rounds=50, verbose=False)
        p = model.predict_proba(X[val_rows])[:, 1]
        threshold = float(max(grid, key=lambda t: f1_score(y_val, p >= t)))
        results.append({'Experiment': name, **evaluate(p, threshold)})
        probabilities[name] = p
        model.save_model(str(out / f'{name}.cbm'))
        print(name, '| F1:', round(results[-1]['F1'], 4), flush=True)

    # Простое смешивание вероятностей без обучения ещё одной модели.
    for weight in (.25, .5, .75):
        p = weight * probabilities.D2_mean_max.to_numpy() + (1 - weight) * old_probability
        threshold = float(max(grid, key=lambda t: f1_score(y_val, p >= t)))
        name = f'D4_probability_fusion_{weight:.2f}'
        results.append({'Experiment': name, **evaluate(p, threshold)})
        probabilities[name] = p

    table = pd.DataFrame(results).sort_values('F1', ascending=False)
    display(table.round(4))
    table.to_csv(out / 'metrics.csv', index=False)
    probabilities.to_csv(out / 'validation_probabilities.csv', index=False)
    print('Сохранено:', out)
    '''),
    md('## 4. Ошибки по меткам и рендеры'),
    code('''
    best = table.iloc[0]
    p = probabilities[best.Experiment].to_numpy() if best.Experiment != 'Geometry previous' else old_probability
    review = val_df[['item_id', 'lowpoly', 'simple', 'noisy']].copy()
    review['probability'] = p
    review['prediction'] = (p >= best.Threshold).astype(int)
    for label, mask in [('lowpoly + simple', review.lowpoly.eq(1) & review.simple.eq(1)),
                        ('lowpoly + noisy', review.lowpoly.eq(1) & review.noisy.eq(1))]:
        group = review[mask]
        print(label, '| объектов:', len(group), '| пропущено:', int(group.prediction.eq(0).sum()))

    png_dir = ROOT / 'data/train'
    png_zip = ROOT / 'data/train_png.zip'
    for title, frame in [('FP', review.query('lowpoly == 0 and prediction == 1')),
                         ('FN', review.query('lowpoly == 1 and prediction == 0'))]:
        sample = frame.sort_values('probability', ascending=title == 'FN').head(2)
        display(sample)
        for item_id in sample.item_id:
            if (png_dir / f'{item_id}.png').is_file():
                image = Image.open(png_dir / f'{item_id}.png')
            else:
                with zipfile.ZipFile(png_zip) as archive:
                    name = next(name for name in archive.namelist() if name.endswith(f'{item_id}.png'))
                    image = Image.open(io.BytesIO(archive.read(name)))
            plt.figure(figsize=(12, 6))
            plt.imshow(image.convert('RGB'))
            plt.title(f'{title}: {item_id}', color='#0098F8')
            plt.axis('off')
            plt.show()
            image.close()
    '''),
    md('''
    ## Вывод

    Сравниваем лучший F1 с прежней геометрией 0.5984 и общим baseline 0.5464. Результат на validation используется для подбора порога и варианта модели, поэтому итоговая оценка может быть оптимистичной. Internal test не используется. MeshCNN и DGCNN пока не запускаем.
    '''),
]

notebook = nb.v4.new_notebook(cells=cells)
notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                     'language_info': {'name': 'python'}}
path = Path(__file__).with_name('lowpoly_dinov2.ipynb')
nb.write(notebook, path)
print(path)
