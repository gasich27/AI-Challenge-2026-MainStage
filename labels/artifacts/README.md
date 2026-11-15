# artifacts

Статус: в исследовании. Есть результаты, это ещё не финальная модель сервиса.

Локальная геометрия и компоненты пока слабее изображений. Добавление геометрии к DINOv2 дало небольшой прирост F1, но выросло число ложных срабатываний. MIL подготовлен, результата ещё нет.

| Выбранный вариант | F1 validation | F1 internal test |
|---|---:|---:|
| DINOv2 + геометрия | .3717 | N/A |

Пороги и варианты выбирали на validation. Отложенную выборку для подбора не используем. Числа из прошлых запусков сохранены, полное обучение после рефакторинга не повторялось.

## Код

- [artifacts_dinov2.ipynb](artifacts_dinov2.ipynb)
- [artifacts_errors.ipynb](artifacts_errors.ipynb)
- [artifacts_examples.ipynb](artifacts_examples.ipynb)
- [artifacts_geometry.ipynb](artifacts_geometry.ipynb)
- [artifacts_mil.ipynb](artifacts_mil.ipynb)

Данные лежат в `../../data/`, кэши — в `artifacts/`, веса — в `models/`. Они не входят в Git. Порядок запуска и нужные файлы — в [инструкции](../../docs/RUNNING.md).

Все сравнения, включая неудачные опыты, — в [журнале](../../docs/experiments/artifacts.md). [Таблица результатов](../../docs/results.csv) содержит источники; [проверенные метрики](../../docs/verified_metrics.csv) пересчитаны из доступных вероятностей.
