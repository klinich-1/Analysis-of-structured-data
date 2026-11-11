"""Tables for the article from results/runs_*.jsonl.

    python agent/analyze.py                      # local runs in results/
    python agent/analyze.py experiment/results   # the published runs of the article

Errors of the agent are sorted into causes by simple rules, checked by hand on the logs:
- no_answer: the run ended without an ANSWER line (iteration limit or empty message);
- iqr_rounding: the outlier count equals the count under another algebraically equal occupancy
  formula or under exact rational arithmetic (see iqr_variants);
- iqr_retained: the answer is the number of rows kept by the IQR rule, not the number flagged;
- iqr_as_filter: outside the IQR question the agent removed IQR outliers before computing,
  reading "IQR and constant values retained" as an instruction to filter;
- other.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import evaluate  # noqa: E402

MONTHS = ['2026-05', '2026-06', '2026-07', '2026-08', '2026-09']
# Checklist items (evaluate.CHECKLIST) present in each month, from audit/audit_results.json and the data.
APPLICABLE = {'2026-05': [0, 1, 2, 3, 4, 5, 6, 7, 10], '2026-06': list(range(11)),
              '2026-07': [0, 1, 2, 4, 5, 6, 8, 10], '2026-08': [0, 1, 2, 4, 5, 6, 10],
              '2026-09': [0, 1, 2, 3, 4, 5, 6, 10]}


def _quantile(values, p):
    values = sorted(values)
    h = (len(values) - 1) * p
    low = int(h)
    if low + 1 >= len(values):
        return values[low]
    return values[low] + (values[low + 1] - values[low]) * (h - low)


def iqr_variants(month: str, package: Path = evaluate.PACKAGE) -> dict:
    """Outlier counts under three equal formulas of occupancy in floating point and in exact arithmetic."""
    df = pd.read_csv(evaluate.csv_for_month(month, package))
    d = df[(df.freeParking >= 0) & (df.freeParking <= df.parkingSpaces) & (df.parkingSpaces > 0)]
    out = Counter()
    for _, g in d.groupby('idParking'):
        f, c = g.freeParking, g.parkingSpaces
        for name, occ in (('one_minus', 1 - f / c), ('c_minus_f', (c - f) / c)):
            q1, q3 = occ.quantile(.25), occ.quantile(.75)
            out[name] += int(((occ < q1 - 1.5 * (q3 - q1)) | (occ > q3 + 1.5 * (q3 - q1))).sum())
        exact = [1 - Fraction(int(a), int(b)) for a, b in zip(f, c)]
        q1, q3 = _quantile(exact, Fraction(1, 4)), _quantile(exact, Fraction(3, 4))
        out['exact'] += sum(x < q1 - Fraction(3, 2) * (q3 - q1) or x > q3 + Fraction(3, 2) * (q3 - q1) for x in exact)
    return dict(out)


def load_runs(results: Path) -> list[dict]:
    runs = []
    for path in sorted(results.glob('runs_*.jsonl')):
        runs += [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    return [r for r in runs if not r.get('error')]


def _code(run) -> str:
    return '\n'.join(json.loads(c['function']['arguments']).get('code', '')
                     for m in run.get('messages', []) if m['role'] == 'assistant' for c in (m.get('tool_calls') or []))


def cause(run, variants) -> str:
    answer = evaluate.extract_answer(run.get('raw_answer'))
    if answer is None:
        return 'no_answer'
    if run['metric'] == 'iqr_flag_n':
        digits = re.search(r'\d+', answer.replace(' ', ''))
        value = int(digits.group()) if digits else None
        if value is not None and value > 10000:
            return 'iqr_retained'
        if value in set(variants[run['month']].values()):
            return 'iqr_rounding'
        return 'other'
    if 'iqr' in _code(run).lower() or 'quantile(0.75)' in _code(run):
        return 'iqr_as_filter'
    return 'other'


def main() -> None:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results'
    runs = load_runs(results)
    variants = {m: iqr_variants(m) for m in MONTHS}
    references = {(r['month'], r['metric']): r['expected_value'] for r in evaluate.load_references()}
    report = {'iqr_variants': variants, 'iqr_reference': {m: references[(m, 'iqr_flag_n')] for m in MONTHS}}

    accuracy = defaultdict(dict)
    causes = defaultdict(Counter)
    usage = defaultdict(dict)
    for (model, mode, condition), group in _groups(runs, 'questions').items():
        n = len(group)
        key = f'{model}|{mode}|{condition}'
        accuracy[key] = {rule: sum(bool(r[f'correct_{rule}']) for r in group) for rule in ('strict', 'protocol', 'with_p1s')}
        accuracy[key]['n'] = n
        accuracy[key]['without_iqr'] = sum(bool(r['correct_protocol']) for r in group if r['metric'] != 'iqr_flag_n')
        usage[key] = {'tokens': round(sum(r['prompt_tokens'] + r['completion_tokens'] for r in group) / n),
                      'seconds': round(sum(r['seconds'] for r in group) / n, 1),
                      'tool_calls': round(sum(r['tool_calls'] for r in group) / n, 1),
                      'no_answer': sum(evaluate.extract_answer(r.get('raw_answer')) is None for r in group)}
        if mode == 'agent':
            for r in group:
                if not r['correct_protocol']:
                    causes[key][cause(r, variants)] += 1
    for (model, mode, condition), group in _groups(runs, 'defects').items():
        usage[f'{model}|defects'] = {'tokens': round(sum(r['prompt_tokens'] + r['completion_tokens'] for r in group) / len(group)),
                                     'seconds': round(sum(r['seconds'] for r in group) / len(group), 1),
                                     'tool_calls': round(sum(r['tool_calls'] for r in group) / len(group), 1),
                                     'reports': sum(bool((r.get('raw_answer') or '').strip()) for r in group)}
    llm_correct = Counter(r['metric'] for r in runs if r['task'] == 'questions' and r['mode'] == 'llm' and r['correct_protocol'])
    report.update(accuracy=accuracy, causes={k: dict(v) for k, v in causes.items()}, usage=usage, llm_correct_metrics=dict(llm_correct))
    (results / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=1))


def _groups(runs, task):
    groups = defaultdict(list)
    for r in runs:
        if r['task'] == task:
            groups[(r['model'], r['mode'], r['condition'])].append(r)
    return groups


if __name__ == '__main__':
    main()
