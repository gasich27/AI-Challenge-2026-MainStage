# Предыдущие эксперименты

Здесь сохранены общий EDA, ResNet18, PointNet++, CatBoost и связка ResNet18 + CatBoost.

Ноутбуки, их выводы, модели и результаты оставлены для сравнения с новыми экспериментами. При рефакторинге добавлены общее оформление и проверки, убраны комментарии из кода. Алгоритмы обучения сохранены.

- `eda.ipynb` — общий анализ данных.
- `cnn_baseline.ipynb` — ResNet18, анализ ошибок, thresholds и fine-tuning.
- `pointnetpp_baseline.ipynb` — baseline на вершинах.
- `catboost_mesh_baseline.ipynb` — CatBoost на mesh-признаках.
- `cnn_mesh_fusion.ipynb` — смешивание вероятностей ResNet18 и CatBoost.
- `models/` — сохранившиеся обученные модели, локально.
- `artifacts/` — признаки, настройки и thresholds, локально.

В ноутбуках сохранены выводы прежних запусков. После рефакторинга обучение не повторялось. Лучший сохранённый общий score на validation — 0.6284 у ResNet18 + CatBoost с 44 mesh-признаками. Последний PointNet++ получил 0.3698. Это score в масштабе 0–1.

Fusion объединяет вероятности улучшенной ResNet18 и CatBoost.

Для отдельной Improved ResNet18 итоговые пороги лежат в `artifacts/resnet/resnet18_improved_thresholds.json`. В checkpoint остались пороги baseline, с которыми выбиралась лучшая эпоха; финальные пороги подбираются позже. Для итоговой связки используем её собственные пороги из `artifacts/resnet_catboost_features_config.json`.

Новые эксперименты ведём в `../labels/`.
