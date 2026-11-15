import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from catboost import CatBoostClassifier, Pool
from src.data import align_features, split_data
from src.plots import setup_plot_style, save_figure


def main():
    labels = pd.read_csv(ROOT / 'data/train.csv')
    train, val, _ = split_data(labels)
    features = pd.read_csv(ROOT / 'archive/artifacts/train_mesh_features_relative.csv').set_index('item_id')
    for name in ['labels/lowpoly/artifacts/geometry_features.csv',
                 'labels/noisy/artifacts/smoothing_features.csv',
                 'labels/artifacts/artifacts/local_edge_features.csv',
                 'labels/artifacts/artifacts/component_features.csv',
                 'labels/simple/geometry_diagnostics_v1.csv']:
        extra = pd.read_csv(ROOT / name).set_index('item_id')
        if not extra.index.is_unique:
            raise ValueError(name)
        columns = [c for c in extra if c not in features and pd.api.types.is_numeric_dtype(extra[c])]
        features = features.join(extra[columns])
    jobs = [
        ('simple', 'simple_catboost_49.cbm', '49 features', 'models/simple_catboost_geometry_validation.csv', 'geometry_probability'),
        ('lowpoly', 'A_no_curvature.cbm', 'A_no_curvature', 'artifacts/validation_probabilities.csv', 'A_no_curvature'),
        ('noisy', 'E4_all.cbm', 'E4_all', 'artifacts/validation_probabilities.csv', 'E4_all'),
        ('artifacts', 'reused_local_components.cbm', 'reused_local_components', 'artifacts/geometry_validation.csv', 'reused_local_components'),
    ]
    verified = pd.read_csv(ROOT / 'docs/verified_metrics.csv')
    rows = []
    setup_plot_style()
    for label, filename, experiment, prediction_file, column in jobs:
        model = CatBoostClassifier().load_model(str(ROOT / 'labels' / label / 'models' / filename))
        columns = model.feature_names_
        X = align_features(features[columns], val.item_id)
        saved = align_features(pd.read_csv(ROOT/'labels'/label/prediction_file), val.item_id)[column].to_numpy()
        current = model.predict_proba(X)[:, 1]
        if not np.allclose(saved, current, atol=1e-10):
            raise ValueError(f'{label}: модель и сохранённые вероятности расходятся')
        sample = X.sample(min(128, len(X)), random_state=42)
        shap = model.get_feature_importance(Pool(sample), type='ShapValues', thread_count=4)[:, :-1]
        table = pd.DataFrame({'feature': columns, 'mean_abs_shap': np.abs(shap).mean(axis=0),
                              'importance': model.get_feature_importance()}).sort_values('mean_abs_shap', ascending=False)
        table['label'] = label
        table['experiment'] = experiment
        table['sample_size'] = len(sample)
        rows.extend(table.to_dict('records'))
        top = table.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top.feature, top.mean_abs_shap)
        ax.set(xlabel='Mean |SHAP|, raw score', title=f'{label}: {experiment}, 128 validation objects')
        fig.tight_layout()
        save_figure(fig, ROOT/'docs/figures'/f'{label}_shap')
        plt.close(fig)
        strongest = table.feature.iloc[0]
        correlations = features.loc[train.item_id, columns].corr(method='spearman')[strongest].drop(strongest)
        correlations.abs().nlargest(10).rename('abs_spearman').to_csv(ROOT/'docs'/f'{label}_correlations.csv')
        fig, ax = plt.subplots(figsize=(7, 4))
        values = [features.loc[train.loc[train[label].eq(target), 'item_id'], strongest].dropna().to_numpy() for target in [0, 1]]
        ax.boxplot(values, showfliers=False)
        ax.set_xticks([1, 2], ['0', '1'])
        ax.set(xlabel=label, ylabel=strongest, title='Train: распределение признака')
        save_figure(fig, ROOT/'docs/figures'/f'{label}_feature')
        plt.close(fig)
        threshold = verified.loc[(verified.label == label) & (verified.Experiment == experiment), 'Threshold'].iloc[0]
        review = val[['item_id', label]].copy()
        review['p'] = saved
        review['prediction'] = (saved >= threshold).astype(int)
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for row_index, kind in enumerate(['FP', 'FN']):
            mask = (review[label] == (kind == 'FN')) & (review.prediction == (kind == 'FP'))
            cases = review[mask].sort_values('p', ascending=kind == 'FN').head(3)
            for ax, (_, row) in zip(axes[row_index], cases.iterrows()):
                with Image.open(ROOT/'data/train'/f'{row.item_id}.png') as image:
                    ax.imshow(image)
                ax.set_title(f'{kind}: {row.item_id[:8]} | p={row.p:.3f}')
                ax.axis('off')
        fig.suptitle(f'{label}: геометрическая модель, реальные ошибки validation')
        fig.tight_layout()
        save_figure(fig, ROOT/'docs/figures'/f'{label}_errors')
        plt.close(fig)
        print(label, 'predictions match; SHAP and errors saved', flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'docs/feature_importance.csv', index=False)


if __name__ == '__main__':
    main()
