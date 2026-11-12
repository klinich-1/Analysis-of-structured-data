# Агент и оценка по эталонам

| Файл | Что делает |
| --- | --- |
| `csv_agent.py` | Агент: отвечает на вопрос по CSV, сам пишет и запускает код на pandas |
| `evaluate.py` | Прогоняет LLM и агента по 85 эталонным вопросам и проверке качества данных, считает результаты |
| `run_experiment.py` | Полный эксперимент: три модели параллельно с общим лимитом токенов |
| `analyze.py` | Таблицы статьи по результатам запусков |
| `test_agent.py` | Тесты без обращения к API |
| `csv_agent.ipynb` | Те же запуски в Google Colab |

## Подготовка

Из корня репозитория:

```sh
pip install -r agent/requirements.txt
python -c "import zipfile; zipfile.ZipFile('data/parking_team_package_no_pdf.zip').extractall()"
```

Ключ задайте в переменной окружения и не сохраняйте его в файлах. В Colab используйте Secrets.

- OpenRouter: `OPENROUTER_API_KEY`.
- Любой другой сервис, совместимый с OpenAI: `LLM_BASE_URL` (адрес вида `https://.../v1`) и `LLM_API_KEY`.

## Запуск

Один вопрос:

```sh
python agent/csv_agent.py parking_team_package/data/parking_places_hourly_202605.csv "Сколько строк с freeParking = -1?"
```

Эксперимент: сначала посмотрите план, потом проверьте на трёх вопросах и только после этого запускайте всё.

```sh
python agent/evaluate.py questions --mode agent --condition protocol --dry-run
python agent/evaluate.py questions --mode agent --condition protocol --limit 3
python agent/evaluate.py questions --mode agent --condition protocol
python agent/evaluate.py summary
```

Параметры:

- `--mode agent` или `llm`: агент с инструментами или модель без инструментов, которая видит только первые `--sample-rows` строк;
- `--condition package` или `protocol`: вопрос как в пакете или с явными соглашениями из статьи;
- `--model`: идентификатор модели в OpenRouter;
- `--months`, `--metrics`, `--limit`: часть вопросов.

Проверка качества данных:

```sh
python agent/evaluate.py defects --months 2026-06
```

Для этой задачи агенту дается до 20 запросов, а если они закончились — еще один запрос без инструментов для итогового отчета. Команда сохранит ответы агента и создаст `results/defect_annotation.csv`. Два человека независимо отмечают в нём, какие дефекты из контрольного списка найдены.

## Результаты

`results/runs.jsonl` хранит каждый запуск: запрос, ответ, вызовы инструментов, токены, время. Завершённые запуски при повторе пропускаются, запуски с ошибкой повторяются. `results/summary.csv` содержит долю верных ответов по трём правилам:

- `strict`: допуски пакета;
- `protocol`: правила статьи;
- `with_p1s`: правила статьи, а для занятости и самой загруженной парковки засчитывается и ответ по P1-S.

## Безопасность

Код модели выполняется в отдельном процессе с ограничением времени, во временной папке и без ключей в окружении. Это не полноценная песочница: запускайте агента в Colab или контейнере.
