# noisy

Статус: в исследовании. Есть результаты, это ещё не финальная модель сервиса.

Сглаживание вместе с локальной геометрией помогло. Конкатенация шести ракурсов не улучшила validation. DINOv3 не запускался: нет доступа к весам. На internal test визуальная модель дала .8429, но менять победителя по test не будем.

| Выбранный вариант | F1 validation | F1 internal test |
|---|---:|---:|
| DINOv2 + геометрия | .8301 | .8385 |

Пороги и варианты выбирали на validation. Отложенную выборку для подбора не используем. Числа из прошлых запусков сохранены, полное обучение после рефакторинга не повторялось.

## Код

- [noisy_dinov2.ipynb](noisy_dinov2.ipynb)
- [noisy_dinov2_test.ipynb](noisy_dinov2_test.ipynb)
- [noisy_dinov3.ipynb](noisy_dinov3.ipynb)
- [noisy_geometry.ipynb](noisy_geometry.ipynb)
- [noisy_six_views.ipynb](noisy_six_views.ipynb)

Данные лежат в `../../data/`, кэши — в `artifacts/`, веса — в `models/`. Они не входят в Git. Порядок запуска и нужные файлы — в [инструкции](../../docs/RUNNING.md).

Все сравнения, включая неудачные опыты, — в [журнале](../../docs/experiments/noisy.md). [Таблица результатов](../../docs/results.csv) содержит источники; [проверенные метрики](../../docs/verified_metrics.csv) пересчитаны из доступных вероятностей.
