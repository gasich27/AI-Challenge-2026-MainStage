import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.data import align_features, split_data
from src.evaluation import binary_metrics
from src.plots import evaluation_plot, save_figure, setup_plot_style


def main():
    labels = pd.read_csv(ROOT / 'data/train.csv')
    _, val, _ = split_data(labels)
    jobs = [
        ('lowpoly', 'artifacts/metrics.csv', 'artifacts/validation_probabilities.csv'),
        ('noisy', 'artifacts/experiments.csv', 'artifacts/validation_probabilities.csv'),
        ('noisy', 'models/six_views/metrics.csv', 'models/six_views/validation_probabilities.csv'),
        ('artifacts', 'artifacts/geometry_metrics.csv', 'artifacts/geometry_validation.csv'),
        ('artifacts', 'models/dinov2/metrics.csv', 'models/dinov2/validation_probabilities.csv'),
        ('simple', 'models/simple_catboost_geometry_results.csv', 'models/simple_catboost_geometry_validation.csv'),
    ]
    rows = []
    for label, metrics_file, predictions_file in jobs:
        folder = ROOT / 'labels' / label
        if not (folder / predictions_file).is_file() or not (folder / metrics_file).is_file():
            print('Нет локальных файлов:', label, predictions_file)
            continue
        metrics = pd.read_csv(folder / metrics_file).rename(columns={
            'Model': 'Experiment', 'model': 'Experiment', 'threshold': 'Threshold'})
        predictions = align_features(pd.read_csv(folder / predictions_file), val.item_id)
        y = val[label].to_numpy()
        if label in predictions and not np.array_equal(y, predictions[label].to_numpy()):
            raise ValueError(f'{label}: разметка сохранённых вероятностей не совпала')
        for _, row in metrics.iterrows():
            name = row.Experiment
            column = {'44 features': 'baseline_probability', '49 features': 'geometry_probability'}.get(name, name)
            p = predictions[column].to_numpy()
            result = binary_metrics(y, p, row.Threshold)
            if not np.isclose(result['F1'], row.F1, atol=1e-10):
                raise ValueError(f'{label}/{name}: F1 не воспроизвёлся')
            rows.append({'label': label, 'Experiment': name, 'split': 'validation', **result,
                         'source': f'labels/{label}/{predictions_file}'})
        best = metrics.loc[metrics.F1.idxmax()]
        column = {'44 features': 'baseline_probability', '49 features': 'geometry_probability'}.get(best.Experiment, best.Experiment)
        fig = evaluation_plot(y, predictions[column].to_numpy(), best.Threshold, f'{label}: {best.Experiment}')
        suffix = Path(metrics_file).parent.name
        save_figure(fig, ROOT / 'docs/figures' / f'{label}_{suffix}_evaluation')
        plt.close(fig)
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / 'docs/verified_metrics.csv', index=False)
    setup_plot_style()
    for label, group in result.groupby('label', sort=False):
        group = group.sort_values('F1')
        fig, axes = plt.subplots(1, 2, figsize=(12, max(3, len(group) * .45)))
        axes[0].barh(group.Experiment, group.F1, label='F1')
        axes[0].scatter(group.AP, group.Experiment, color='#21A038', label='AP', zorder=3)
        axes[0].set(xlim=(0, 1), xlabel='Validation', title=label)
        axes[0].legend()
        axes[1].barh(group.Experiment, group.FP, label='FP', color='#0098F8')
        axes[1].barh(group.Experiment, group.FN, left=group.FP, label='FN', color='#F2C614')
        axes[1].set(xlabel='Ошибки', title='FP + FN')
        axes[1].legend()
        fig.tight_layout()
        save_figure(fig, ROOT / 'docs/figures' / f'{label}_comparison')
        plt.close(fig)
    print('Проверены метрики:', len(result), '| графики: docs/figures')


if __name__ == '__main__':
    main()
