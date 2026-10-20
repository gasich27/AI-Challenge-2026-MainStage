# AI Challenge 2026 — 3D Mesh Quality

Решение основного этапа AI Challenge 2026 от SBER AI.

## Задача

Нужно по 3D-модели определить:

- хорошая она или плохая;
- какие дефекты у неё есть.

У одного объекта может быть сразу несколько дефектов, поэтому это multi-label классификация.

Всего есть 10 дефектов:

- `abstract`
- `artifacts`
- `intersection`
- `lowpoly`
- `noisy`
- `open`
- `partial`
- `scale`
- `set`
- `simple`

Также отдельно есть метка `quality`.


## Данные

Для каждого объекта есть:

- PNG с шестью ракурсами модели;
- NPZ с вершинами и гранями mesh.

В train находится 8 964 объекта.


## Метрика

Основная метрика состоит из:

- F1 для `quality`;
- weighted F1 для дефектов.

---
---

## EDA

EDA находится в [`eda.ipynb`](eda.ipynb).

Там посмотрел:

- пропуски и дубликаты;
- баланс классов;
- распределение `quality`;
- частоту дефектов;
- корреляции между дефектами;
- примеры хороших и плохих объектов;
- количество вершин и граней mesh.

---
---

## ResNet18 baseline

Основной ноутбук находится в [`cnn_baseline.ipynb`](cnn_baseline.ipynb).

Первый baseline обучается только на PNG:

```text
PNG
↓
ResNet18
↓
10 дефектов + quality
```

Используется:

- pretrained ResNet18;
- размер изображения `224 × 336`;
- `BCEWithLogitsLoss`;
- веса для редких классов;
- 3 эпохи обучения;
- сохранение лучшей модели.

Данные разделены на:

- train — обучение;
- validation — подбор модели и анализ ошибок;
- internal test — финальная проверка.

Internal test пока не используется.

---
---

## Результат baseline

На validation получилось:

| Метрика | Значение |
|---|---:|
| Macro F1 | 0.4337 |
| Weighted F1 дефектов | 0.5854 |
| F1 quality | 0.5225 |
| Score | 0.5539 |

---
---

## Анализ ошибок

Для каждого класса посчитал:

- precision;
- recall;
- F1;
- TP, FP, FN и TN.

Самые сложные классы:

- `intersection`
- `scale`
- `artifacts`
- `open`
- `lowpoly`

Для них отдельно вывожу false positive и false negative с исходными изображениями.

---
---

## Threshold tuning

Изначально для всех классов использовался threshold `0.5`.

После этого подобрал отдельный threshold для каждого класса на validation.

Результат:

| Метрика | Было | Стало |
|---|---:|---:|
| Macro F1 | 0.4337 | 0.4876 |
| Weighted F1 дефектов | 0.5854 | 0.6105 |
| F1 quality | 0.5225 | 0.5514 |
| Score | 0.5539 | 0.5810 |

Подобранные пороги сохраняются в `artifacts/resnet/resnet18_thresholds.json`.

---
---

## Улучшение ResNet18

Сейчас добавил простые аугментации:

- горизонтальное отражение;
- небольшое изменение яркости и контраста;
- небольшое изменение насыщенности;
- редкое преобразование в оттенки серого.

Также добавил частичный fine-tuning:

- ранние слои ResNet18 заморожены;
- обучается последний блок `layer4`;
- обучается классификатор `fc`;
- используется небольшой learning rate.

Лучшая улучшенная модель сохраняется в `models/resnet/best_resnet18_improved.pth`.

После fine-tuning ещё раз подобрал thresholds уже для улучшенной модели.

Итоговый результат ResNet18:

| Метрика | Baseline | Improved ResNet18 |
|---|---:|---:|
| Macro F1 | 0.4876 | 0.5069 |
| Weighted F1 дефектов | 0.6105 | 0.6204 |
| F1 quality | 0.5514 | 0.5708 |
| Score | 0.5810 | 0.5956 |

Новые thresholds сохраняются в `artifacts/resnet/resnet18_improved_thresholds.json`.

---
---

## PointNet++ baseline

Baseline на геометрии находится в [`pointnetpp_baseline.ipynb`](pointnetpp_baseline.ipynb).

Для него используются только вершины из NPZ:

```text
NPZ
↓
512 вершин
↓
PointNet++
↓
10 дефектов + quality
```

Для каждого объекта:

- выбираются 512 вершин;
- точки центрируются;
- координаты нормализуются;
- локальные признаки собираются двумя слоями Set Abstraction;
- модель предсказывает 11 меток.

Используется то же разбиение на train, validation и internal test, что и для ResNet18.

Модель обучалась 5 эпох с `BCEWithLogitsLoss` и весами для редких классов.

Лучший результат получился на 4 эпохе:

| Метрика | Значение |
|---|---:|
| Macro F1 | 0.2658 |
| Weighted F1 дефектов | 0.4200 |
| F1 quality | 0.3260 |
| Score | 0.3730 |

Лучший checkpoint сохраняется в `models/pointnetpp/best_pointnetpp.pth`.

PointNet++ получает полезный сигнал только из вершин, но работает слабее модели по изображениям.

---
---

## CatBoost mesh baseline

Baseline на простых признаках mesh находится в [`catboost_mesh_baseline.ipynb`](catboost_mesh_baseline.ipynb).

Из NPZ извлекаются 27 признаков:

- количество вершин и граней;
- размеры bounding box;
- пропорции объекта;
- статистики координат;
- длины рёбер;
- площади треугольников.

Для каждой метки обучается отдельный `CatBoostClassifier`.

Результат baseline с threshold `0.5`:

| Метрика | Значение |
|---|---:|
| Macro F1 | 0.3642 |
| Weighted F1 дефектов | 0.5364 |
| F1 quality | 0.5009 |
| Score | 0.5186 |

CatBoost работает заметно лучше PointNet++. Следующий эксперимент — подбор его параметров на validation.

Признаки сохраняются в `artifacts/train_mesh_features.csv`, а модели — в `models/catboost`.

---
---

## Структура проекта

```text
eda.ipynb
cnn_baseline.ipynb
pointnetpp_baseline.ipynb
catboost_mesh_baseline.ipynb
cnn_mesh_fusion.ipynb

data/
├── train.csv
└── train/

models/
├── resnet/
├── pointnetpp/
├── catboost/
└── catboost_tuned/

artifacts/
├── resnet/
├── train_mesh_features.csv
└── настройки моделей и thresholds

assets/
└── Frame 768.png
```

---
---
---

![alt text](<assets/Frame 768.png>)
