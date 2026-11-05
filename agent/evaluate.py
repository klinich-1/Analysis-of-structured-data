"""Compare a baseline LLM and the agent on the 85 reference questions and on defect detection.

Examples (from the repository root, after unpacking data/parking_team_package_no_pdf.zip):

    python agent/evaluate.py questions --dry-run
    python agent/evaluate.py questions --mode agent --condition protocol --model anthropic/claude-opus-5-5 --limit 3
    python agent/evaluate.py defects --model anthropic/claude-opus-5-5
    python agent/evaluate.py summary

Every run is appended to results/runs.jsonl with the full transcript. Finished runs are skipped on restart.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csv_agent import DEFAULT_MODEL, SYSTEM_PROMPT, ask_model, make_client, run_agent  # noqa: E402

PACKAGE = Path('parking_team_package')
PROTOCOL = """Соглашения для ответа:
- фильтр P1: строка допустима, если 0 <= freeParking <= parkingSpaces и parkingSpaces > 0; нули, выбросы и постоянные значения сохраняются;
- стандартное отклонение выборочное, ddof = 1;
- квантили, в том числе для выбросов в группах: линейная интерполяция, h = (n - 1) * p;
- при совпадении значений верни все варианты через точку с запятой;
- доли записывай дробью от 0 до 1, дробные ответы округляй до четырех знаков.
Если данные содержат дефект, влияющий на ответ, ответь по правилу P1 и укажи дефект в тексте перед последней строкой."""
LLM_FORMAT = 'Последняя строка ответа должна иметь вид: ANSWER: <значение>. Несколько значений разделяй точкой с запятой.'
DEFECT_PROMPT = ('Проверь качество данных в этом файле и перечисли все найденные дефекты измерений и справочных полей. '
                 'Для каждого дефекта укажи парковки, масштаб и влияние на средние показатели. Числа получай расчетом.')
# Checklist for two annotators, taken from Tables 3 and 5 of the article.
CHECKLIST = ['Служебное значение -1 в freeParking', 'Парковки, у которых нет допустимых значений',
             'Число свободных мест больше вместимости (ID 32)', 'Значения меньше -1 (ID 37)',
             'Постоянный ноль у ID 15', 'Постоянное значение у ID 49', 'Застывшие отрезки и скачки у ID 80',
             'Застывшие отрезки у ID 76', 'Застывший отрезок у ID 79', 'Застывший отрезок у ID 9',
             'Общесистемные сбои (-1 одновременно у многих парковок)']
FRACTION_TOLERANCE = 2e-4


def csv_for_month(month: str, package: Path = PACKAGE) -> Path:
    return package / 'data' / f"parking_places_hourly_{month.replace('-', '')}.csv"


def load_references(package: Path = PACKAGE) -> list[dict]:
    with open(package / 'tables' / 'reference_answers.csv', encoding='utf-8-sig', newline='') as fh:
        return list(csv.DictReader(fh))


def load_alternatives(path: Path = Path('audit/audit_results.json')) -> dict:
    """References under the P1-S rule, accepted for measures that frozen sensors distort."""
    if not path.exists():
        return {}
    months = dict(zip(['May', 'June', 'July', 'August', 'September'], [f'2026-0{i}' for i in range(5, 10)]))
    data = json.loads(path.read_text(encoding='utf-8'))['monthly']
    out = {}
    for name, month in months.items():
        out[month, 'occupancy_mean'] = str(data[name]['s_rows'])
        out[month, 'occupancy_capacity_weighted'] = str(data[name]['s_w'])
        out[month, 'top_parking_all_ties'] = ';'.join(str(x) for x in data[name]['s_top'])
    return out


def build_prompt(reference: dict, condition: str) -> str:
    if condition == 'package':
        return reference['question_ru']
    if condition == 'protocol':
        return f"{PROTOCOL}\n\nВопрос: {reference['question_ru']}"
    raise ValueError(condition)


def llm_prompt(question: str, csv_path: Path, rows: int) -> str:
    with open(csv_path, encoding='utf-8') as fh:
        lines = [line.rstrip('\n') for line in fh if line.strip()]
    sample = '\n'.join(lines[:rows + 1])
    return (f'{question}\n\nФайл {csv_path.name} содержит {len(lines) - 1} строк данных. '
            f'Ниже заголовок и первые {min(rows, len(lines) - 1)} строк. Инструментов у тебя нет.\n\n{sample}\n\n{LLM_FORMAT}')


def extract_answer(text: str | None) -> str | None:
    if not text:
        return None
    found = re.findall(r'ANSWER:\s*(.+)', text.replace('**', ''))
    return found[-1].strip().strip('`') if found else None


def parse_number(text: str) -> tuple[float, bool] | None:
    """Return (value, written_as_percent) or None. Accepts a decimal comma and spaces in thousands."""
    match = re.fullmatch(r'\s*([+-]?[\d\s ]*\d(?:[.,]\d+)?(?:[eE][+-]?\d+)?)\s*(%?)\s*\.?\s*', text)
    if not match:
        return None
    return float(re.sub(r'[\s ]', '', match.group(1)).replace(',', '.')), bool(match.group(2))


def parse_set(text: str) -> set[str]:
    items = (x.strip() for x in re.split(r'[;,]', text))
    return {re.sub(r'\.0+$', '', x) for x in items if x}


def is_set_metric(metric: str) -> bool:
    return metric.endswith('_all_ties')


def score_strict(reference: dict, answer: str | None) -> bool:
    """Package rule: exact sets and integers, absolute tolerance for fractions."""
    if answer is None:
        return False
    expected = reference['expected_value']
    if is_set_metric(reference['metric']):
        return parse_set(answer) == parse_set(expected)
    got = parse_number(answer)
    if got is None or got[1]:
        return False
    return abs(got[0] - float(expected)) <= float(reference['absolute_tolerance'])


def _fraction_close(value: float, expected: float) -> bool:
    if round(value, 4) == round(expected, 4):
        return True
    return expected != 0 and abs(value - expected) / abs(expected) <= FRACTION_TOLERANCE


def score_protocol(reference: dict, answer: str | None, expected: str | None = None) -> bool:
    """Article protocol: sets and integers exact; fractions rounded or within a relative tolerance; shares in either form."""
    if answer is None:
        return False
    expected = reference['expected_value'] if expected is None else expected
    if is_set_metric(reference['metric']):
        return parse_set(answer) == parse_set(expected)
    got = parse_number(answer)
    if got is None:
        return False
    value, percent = got
    if float(reference['absolute_tolerance']) == 0:
        return not percent and math.isclose(value, float(expected), abs_tol=1e-9)
    if reference['unit'] == 'fraction_0_1':
        if percent or value > 1.5:
            value /= 100
    elif percent:
        return False
    return _fraction_close(value, float(expected))


def score_with_alternative(reference: dict, answer: str | None, alternatives: dict) -> bool:
    if score_protocol(reference, answer):
        return True
    alternative = alternatives.get((reference['month'], reference['metric']))
    return alternative is not None and score_protocol(reference, answer, alternative)


def load_done(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            if r.get('error') is None:
                done.add((r['task'], r['mode'], r['model'], r['condition'], r['month'], r['metric']))
    return done


def _record(result, **fields) -> dict:
    return {**fields, 'stop_reason': result.stop_reason, 'iterations': result.iterations, 'tool_calls': result.tool_calls,
            'prompt_tokens': result.prompt_tokens, 'completion_tokens': result.completion_tokens,
            'seconds': round(result.seconds, 2), 'raw_answer': result.answer, 'messages': result.messages, 'error': None}


def run_questions(args, client=None) -> int:
    references = load_references(args.package)
    if args.months:
        references = [r for r in references if r['month'] in args.months]
    if args.metrics:
        references = [r for r in references if r['metric'] in args.metrics]
    done = load_done(args.out)
    todo = [r for r in references if ('questions', args.mode, args.model, args.condition, r['month'], r['metric']) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f'К запуску: {len(todo)} из {len(references)} вопросов; режим {args.mode}, условие {args.condition}, модель {args.model}.')
    if args.dry_run:
        if todo:
            print('\nПример запроса:\n' + build_prompt(todo[0], args.condition))
        return len(todo)
    client = client or make_client()
    alternatives = load_alternatives()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for i, reference in enumerate(todo, 1):
        path = csv_for_month(reference['month'], args.package)
        prompt = build_prompt(reference, args.condition)
        key = dict(task='questions', mode=args.mode, model=args.model, condition=args.condition,
                   month=reference['month'], metric=reference['metric'])
        try:
            if args.mode == 'agent':
                result = run_agent(client, prompt, path, model=args.model)
            else:
                result = ask_model(client, llm_prompt(prompt, path, args.sample_rows), model=args.model)
            answer = extract_answer(result.answer)
            record = _record(result, **key, expected=reference['expected_value'], answer=answer,
                             correct_strict=score_strict(reference, answer), correct_protocol=score_protocol(reference, answer),
                             correct_with_p1s=score_with_alternative(reference, answer, alternatives))
        except Exception as error:  # keep going: failed runs are recorded and repeated on restart
            record = {**key, 'error': f'{type(error).__name__}: {error}'}
        with open(args.out, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"[{i}/{len(todo)}] {reference['month']} {reference['metric']}: "
              f"{record.get('error') or ('верно' if record['correct_protocol'] else 'неверно')}", flush=True)
    return len(todo)


def run_defects(args, client=None) -> int:
    months = args.months or sorted({r['month'] for r in load_references(args.package)})
    done = load_done(args.out)
    todo = [m for m in months if ('defects', 'agent', args.model, 'open', m, 'defects') not in done]
    print(f'К запуску: {len(todo)} проверок качества данных; модель {args.model}.')
    if args.dry_run:
        return len(todo)
    client = client or make_client()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet = args.out.parent / 'defect_annotation.csv'
    new_sheet = not sheet.exists()
    for month in todo:
        key = dict(task='defects', mode='agent', model=args.model, condition='open', month=month, metric='defects')
        try:
            system = SYSTEM_PROMPT.split('Последняя строка')[0].strip()
            record = _record(run_agent(client, DEFECT_PROMPT, csv_for_month(month, args.package), model=args.model, system=system), **key)
        except Exception as error:
            record = {**key, 'error': f'{type(error).__name__}: {error}'}
        with open(args.out, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        if record['error'] is None:
            with open(sheet, 'a', encoding='utf-8-sig', newline='') as fh:
                writer = csv.writer(fh)
                if new_sheet:
                    writer.writerow(['model', 'month', 'defect', 'annotator_1', 'annotator_2'])
                    new_sheet = False
                writer.writerows([args.model, month, item, '', ''] for item in CHECKLIST)
        print(month, record['error'] or 'готово', flush=True)
    return len(todo)


def summarize(path: Path) -> list[dict]:
    groups = defaultdict(list)
    errors = defaultdict(int)
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            if r['task'] != 'questions':
                continue
            key = (r['model'], r['mode'], r['condition'])
            if r['error'] is None:
                groups[key].append(r)
            else:
                errors[key] += 1
    rows = []
    for (model, mode, condition), items in sorted(groups.items()):
        n = len(items)
        rows.append(dict(model=model, mode=mode, condition=condition, runs=n, errors=errors[model, mode, condition],
                         strict=round(sum(r['correct_strict'] for r in items) / n, 4),
                         protocol=round(sum(r['correct_protocol'] for r in items) / n, 4),
                         with_p1s=round(sum(r['correct_with_p1s'] for r in items) / n, 4),
                         tool_calls=round(sum(r['tool_calls'] for r in items) / n, 2),
                         tokens=round(sum(r['prompt_tokens'] + r['completion_tokens'] for r in items) / n),
                         seconds=round(sum(r['seconds'] for r in items) / n, 1)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['questions', 'defects', 'summary'])
    parser.add_argument('--mode', choices=['agent', 'llm'], default='agent')
    parser.add_argument('--condition', choices=['package', 'protocol'], default='protocol')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--months', nargs='*', help='например 2026-05 2026-08')
    parser.add_argument('--metrics', nargs='*')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sample-rows', type=int, default=200, help='сколько строк видит LLM без инструментов')
    parser.add_argument('--package', type=Path, default=PACKAGE)
    parser.add_argument('--out', type=Path, default=Path('results/runs.jsonl'))
    parser.add_argument('--dry-run', action='store_true', help='показать план, не вызывая модель')
    args = parser.parse_args()
    if args.command == 'summary':
        rows = summarize(args.out)
        if not rows:
            print('Нет завершенных запусков.')
            return
        target = args.out.parent / 'summary.csv'
        with open(target, 'w', encoding='utf-8-sig', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            print(row)
        print('Сохранено:', target)
    elif args.command == 'questions':
        run_questions(args)
    else:
        run_defects(args)


if __name__ == '__main__':
    main()
