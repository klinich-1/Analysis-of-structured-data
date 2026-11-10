"""Run the full experiment: several models in parallel, with a shared token budget.

    python agent/run_experiment.py --budget 22000000

Each model gets its own process and its own results/runs_<model>.jsonl. Before every step a
process adds up the tokens of all models, weighted by the provider's price multipliers, and
stops when the budget would be exceeded. Finished questions are skipped on restart.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {'claude-sonnet-5-5': 2.0, 'gpt-6.1-sol': 2.0, 'deepseek-v4-pro-0813': 0.5}
STEPS = [('questions', 'agent', 'protocol'), ('questions', 'agent', 'package'),
         ('questions', 'llm', 'protocol'), ('questions', 'llm', 'package'), ('defects', 'agent', 'open')]
MONTHS = ['2026-05', '2026-06', '2026-07', '2026-08', '2026-09']


def spent(results: Path) -> float:
    total = 0.0
    for path in results.glob('runs_*.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            total += (r.get('prompt_tokens', 0) + r.get('completion_tokens', 0)) * MODELS.get(r['model'], 1.0)
    return total


def keep_awake() -> None:
    """Ask Windows not to sleep while this process runs; settings are not changed."""
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def run_model(model: str, budget: float, results: Path) -> None:
    keep_awake()
    out = results / f"runs_{model}.jsonl"
    for task, mode, condition in STEPS:
        for month in MONTHS:  # month by month, so the budget is checked often
            used = spent(results)
            if used >= budget:
                print(f'{model}: бюджет исчерпан ({used:,.0f} из {budget:,.0f}), остановка.', flush=True)
                return
            command = [sys.executable, '-B', str(HERE / 'evaluate.py'), task, '--model', model, '--months', month, '--out', str(out)]
            if task == 'questions':
                command += ['--mode', mode, '--condition', condition]
            print(f'{model}: {task} {mode} {condition} {month}; израсходовано {used:,.0f}', flush=True)
            subprocess.run(command, check=False)
    print(f'{model}: готово.', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--budget', type=float, required=True, help='лимит взвешенных токенов на все модели')
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--model', help='запустить одну модель в этом процессе')
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    if args.model:
        run_model(args.model, args.budget, args.results)
        return
    workers = [subprocess.Popen([sys.executable, '-B', __file__, '--budget', str(args.budget), '--results', str(args.results), '--model', m])
               for m in MODELS]
    for worker in workers:
        worker.wait()
    print('Израсходовано всего:', f'{spent(args.results):,.0f}')


if __name__ == '__main__':
    main()
