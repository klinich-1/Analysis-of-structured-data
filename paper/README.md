# Статья

**Fragile Reference Answers on Real Data: An Audit of the EMT Madrid Open Parking Dataset for Evaluating Language Models and AI Agents**

Рукопись подготовлена для журнала «Discrete and Continuous Models and Applied Computational Science» (РУДН).

| Файл | Что это |
| --- | --- |
| [`article.pdf`](article.pdf) | Собранная статья в оформлении журнала |
| `latex/article.tex` | Текст статьи |
| `latex/cite.bib` | Список литературы |
| `latex/image/` | Рисунки 1 и 2 в форматах png и jpg |
| остальные файлы в `latex/` | Шаблон журнала [yamadharma/dcm-author-template](https://github.com/yamadharma/dcm-author-template), лицензия CC BY 4.0 (`latex/TEMPLATE_LICENSE`) |

## Сборка

Нужны LuaLaTeX и biber (TeX Live или MiKTeX), а также шрифты Source Serif 4, Source Sans 3, Source Code Pro и STIX Two Math. В MiKTeX это пакеты `sourceserif`, `sourcesans`, `sourcecodepro` и `stix2-otf`. Из папки `latex/`:

```sh
lualatex default
biber default
lualatex default
lualatex default
```

Результат появится в файле `default.pdf`. В Overleaf выберите компилятор LuaLaTeX.

## Откуда числа

| Раздел статьи | Источник |
| --- | --- |
| Аудит пакета, таблицы 3–8, рисунки | [`audit/audit_repro.py`](../audit/README.md) |
| Эксперимент с LLM и агентами, таблицы 9–11 | [`agent/analyze.py`](../agent/README.md) по логам в [`experiment/`](../experiment/README.md) |
| Поиск дефектов агентами, таблица 12 | Разметка двух разметчиков в `experiment/results/defect_annotation.csv` и `false_defects.csv` |
