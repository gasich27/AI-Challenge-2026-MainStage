"""Вероятности старой ResNet18 + CatBoost связки на validation."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from catboost import CatBoostClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm.auto import tqdm


root = Path(__file__).resolve().parents[2]
labels = pd.read_csv(root / 'data/train.csv')
targets = labels.columns.drop('item_id').tolist()
split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.15, random_state=42)
train_val_idx, _ = next(split.split(labels, labels[targets]))
train_val = labels.iloc[train_val_idx].reset_index(drop=True)
split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.1765, random_state=43)
_, val_idx = next(split.split(train_val, train_val[targets]))
val = train_val.iloc[val_idx].reset_index(drop=True)
assert len(val) == 1361

transform = transforms.Compose([
    transforms.Resize((224, 336)), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


class Renders(Dataset):
    def __len__(self):
        return len(val)

    def __getitem__(self, index):
        path = root / 'data/train' / f'{val.item_id.iloc[index]}.png'
        with Image.open(path) as image:
            return transform(image.convert('RGB'))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(targets))
checkpoint = torch.load(root / 'archive/models/resnet/best_resnet18_improved.pth',
                        map_location='cpu', weights_only=False)
assert checkpoint['target_columns'] == targets
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device).eval()
probabilities = []
with torch.inference_mode():
    for images in tqdm(DataLoader(Renders(), batch_size=32, num_workers=0), desc='ResNet18'):
        probabilities.append(model(images.to(device)).sigmoid().cpu().numpy())
resnet_p = np.concatenate(probabilities)[:, targets.index('noisy')]

config = json.loads((root / 'archive/artifacts/catboost_features_config.json').read_text(encoding='utf-8'))
features = pd.read_csv(root / 'archive/artifacts/train_mesh_features_relative.csv').set_index('item_id')
catboost = CatBoostClassifier()
catboost.load_model(str(root / 'archive/models/catboost_features/noisy.cbm'))
catboost_p = catboost.predict_proba(features.loc[val.item_id, config['feature_columns']])[:, 1]
fusion_p = .5 * resnet_p + .5 * catboost_p
print('Old fusion F1:', round(f1_score(val.noisy, fusion_p >= .47), 4))
out = root / 'labels/noisy/artifacts/old_fusion_validation.csv'
pd.DataFrame({'item_id': val.item_id, 'noisy': val.noisy,
              'resnet': resnet_p, 'catboost': catboost_p,
              'fusion': fusion_p}).to_csv(out, index=False)
print('Saved:', out)
