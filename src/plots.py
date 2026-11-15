from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from src.evaluation import binary_metrics

COLORS = ['#0098F8', '#21A038', '#F2C614']


def setup_plot_style():
    plt.rcParams.update({
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'text.color': '#1E293B', 'axes.labelcolor': '#1E293B',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#E2E8F0', 'axes.axisbelow': True,
        'axes.prop_cycle': plt.cycler(color=COLORS), 'font.size': 10,
        'figure.dpi': 110, 'svg.fonttype': 'none',
    })


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg']:
        fig.savefig(path.with_suffix('.' + extension), dpi=170, bbox_inches='tight')
    svg = path.with_suffix('.svg')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines()) + '\n', encoding='utf8')


def evaluation_plot(y, p, threshold, title):

    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y, p)
    axes[0, 0].plot(recall, precision)
    axes[0, 0].set(xlabel='Recall', ylabel='Precision', title='PR curve')
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        axes[0, 1].plot(fpr, tpr)
    axes[0, 1].set(xlabel='FPR', ylabel='TPR', title='ROC curve')
    grid = np.arange(.01, 1, .01)
    axes[1, 0].plot(grid, [binary_metrics(y, p, t)['F1'] for t in grid])
    axes[1, 0].axvline(threshold, color=COLORS[2], linestyle='--')
    axes[1, 0].set(xlabel='Threshold', ylabel='F1', title='Validation F1')
    matrix = confusion_matrix(y, np.asarray(p) >= threshold, labels=[0, 1])
    axes[1, 1].imshow(matrix, cmap='Blues')
    for (i, j), value in np.ndenumerate(matrix):
        axes[1, 1].text(j, i, str(value), ha='center', va='center',
                        color='white' if value > matrix.max()/2 else '#1E293B')
    axes[1, 1].grid(False)
    axes[1, 1].set(xticks=[0, 1], yticks=[0, 1], xlabel='Prediction', ylabel='Target', title='Confusion matrix')
    fig.suptitle(title)
    fig.tight_layout()
    return fig
