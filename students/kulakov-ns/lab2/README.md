# Лабораторная работа № 2

## Цель работы
Реализовать метод Random Forest, подобрать гиперпараметры по OOB через `GridSearchCV`, оценить важность признаков через OOB^j и сравнить результат с эталонной реализацией `RandomForestClassifier` из `scikit-learn`.

## Структура проекта
- `main.py` — запуск эксперимента и генерация отчета;
- `models/my_forest.py` — собственная реализация Random Forest;
- `models/sklearn_forest.py` — эталонная реализация на `scikit-learn`;
- `utils/dataset.py` — загрузка и подготовка данных;
- `utils/metrics.py` — расчет метрик и времени обучения;
- `data/report.md` — автоматически сформированный отчет.

## Запуск
```bash
python main.py
```

После запуска будет обновлен файл `data/report.md`.
