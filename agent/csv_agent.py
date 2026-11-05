"""Tool-using agent for questions about one CSV file.

The model reaches any provider model through the OpenAI-compatible OpenRouter API.
It sees the file only through two tools: describe_csv and run_python.

run_python executes model-written code in a separate process with a time limit, a
temporary working directory and an environment without API keys. This is process
isolation, not a security sandbox: run the agent in Colab or a container.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = 'anthropic/claude-opus-5-5'
BASE_URL = 'https://openrouter.ai/api/v1'
SAFE_ENV = ('PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'HOME', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'LANG')
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

SYSTEM_PROMPT = """Ты анализируешь один CSV-файл с помощью инструментов.
Не угадывай значения: вычисляй их кодом и проверяй данные, прежде чем им доверять.
Переменная df в run_python уже содержит таблицу. Каждый вызов выполняется заново, поэтому результат нужно печатать через print.
Последняя строка ответа должна иметь вид: ANSWER: <значение>. Несколько значений разделяй точкой с запятой."""

TOOLS = [
    {'type': 'function', 'function': {
        'name': 'describe_csv',
        'description': 'Возвращает размер таблицы, столбцы, типы, число пропусков и первые строки.',
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}},
    {'type': 'function', 'function': {
        'name': 'run_python',
        'description': 'Выполняет код на Python. Доступны pandas как pd и таблица df. Возвращает напечатанный вывод.',
        'parameters': {'type': 'object', 'properties': {'code': {'type': 'string', 'description': 'Код на Python'}},
                       'required': ['code'], 'additionalProperties': False}}},
]


def run_python(code: str, csv_path: str | Path, timeout: int = 60, max_output: int = 8000) -> str:
    """Run code in a separate process and return its output or error."""
    prelude = f'import pandas as pd\ndf = pd.read_csv({str(Path(csv_path).resolve())!r})\n'
    env = {k: os.environ[k] for k in SAFE_ENV if k in os.environ}
    env['PYTHONIOENCODING'] = 'utf-8'
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / 'job.py'
        script.write_text(prelude + code, encoding='utf-8')
        try:
            done = subprocess.run([sys.executable, '-B', str(script)], cwd=tmp, env=env, capture_output=True,
                                  text=True, encoding='utf-8', errors='replace', timeout=timeout)
        except subprocess.TimeoutExpired:
            return f'Ошибка: код не завершился за {timeout} с.'
    text = done.stdout if done.returncode == 0 else f'Ошибка выполнения:\n{done.stderr}'
    if len(text) > max_output:
        text = text[:max_output] + f'\n[вывод обрезан до {max_output} символов]'
    return text or '(код ничего не напечатал)'


def describe_csv(csv_path: str | Path) -> str:
    code = ("import json\n"
            "print(json.dumps({'shape': df.shape, 'columns': list(df.columns),"
            " 'dtypes': df.dtypes.astype(str).to_dict(), 'missing': df.isna().sum().to_dict(),"
            " 'head': df.head(5).to_dict(orient='records')}, ensure_ascii=False, default=str))\n")
    return run_python(code, csv_path)


@dataclass
class AgentResult:
    answer: str | None
    stop_reason: str
    iterations: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    messages: list = field(default_factory=list)


def make_client(api_key: str | None = None):
    from openai import OpenAI
    key = api_key or os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise SystemExit('Нет ключа: задайте переменную окружения OPENROUTER_API_KEY.')
    return OpenAI(api_key=key, base_url=BASE_URL)


def _create(client, retries: int, **kwargs):
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as error:  # the SDK raises several classes; retry only temporary failures
            if getattr(error, 'status_code', None) not in RETRY_STATUS or attempt == retries:
                raise
            time.sleep(2 ** attempt)


def _call_tool(name: str, raw_args: str, csv_path) -> str:
    try:
        args = json.loads(raw_args or '{}')
    except json.JSONDecodeError:
        return 'Ошибка: аргументы инструмента не являются JSON.'
    if name == 'describe_csv':
        return describe_csv(csv_path)
    if name == 'run_python':
        if not isinstance(args.get('code'), str):
            return 'Ошибка: нужен строковый аргумент code.'
        return run_python(args['code'], csv_path)
    return f'Ошибка: неизвестный инструмент {name}.'


def _usage(result: AgentResult, response) -> None:
    usage = getattr(response, 'usage', None)
    result.prompt_tokens += getattr(usage, 'prompt_tokens', 0) or 0
    result.completion_tokens += getattr(usage, 'completion_tokens', 0) or 0


def ask_model(client, question: str, model: str = DEFAULT_MODEL, max_tokens: int = 2048,
              retries: int = 4, system: str | None = None) -> AgentResult:
    """One request without tools: the baseline LLM."""
    started = time.monotonic()
    messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': question}]
    response = _create(client, retries, model=model, messages=messages, max_tokens=max_tokens, temperature=0)
    choice = response.choices[0]
    result = AgentResult(answer=choice.message.content, stop_reason=choice.finish_reason, iterations=1, messages=messages)
    _usage(result, response)
    result.seconds = time.monotonic() - started
    return result


def run_agent(client, question: str, csv_path: str | Path, model: str = DEFAULT_MODEL, max_iterations: int = 12,
              max_tokens: int = 4096, retries: int = 4, system: str = SYSTEM_PROMPT) -> AgentResult:
    """Run the tool loop until the model answers or a limit is reached."""
    started = time.monotonic()
    messages = [{'role': 'system', 'content': system},
                {'role': 'user', 'content': f'Файл: {Path(csv_path).name}\n\nВопрос: {question}'}]
    result = AgentResult(answer=None, stop_reason='max_iterations', messages=messages)
    for _ in range(max_iterations):
        result.iterations += 1
        response = _create(client, retries, model=model, messages=messages, tools=TOOLS,
                           max_tokens=max_tokens, temperature=0)
        _usage(result, response)
        choice = response.choices[0]
        message = choice.message
        calls = message.tool_calls or []
        if not calls:
            result.answer = message.content
            result.stop_reason = choice.finish_reason
            messages.append({'role': 'assistant', 'content': message.content})
            break
        messages.append({'role': 'assistant', 'content': message.content, 'tool_calls': [
            {'id': c.id, 'type': 'function', 'function': {'name': c.function.name, 'arguments': c.function.arguments}}
            for c in calls]})
        for call in calls:
            result.tool_calls += 1
            output = _call_tool(call.function.name, call.function.arguments, csv_path)
            messages.append({'role': 'tool', 'tool_call_id': call.id, 'content': output})
    result.seconds = time.monotonic() - started
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description='Ответить на вопрос по CSV-файлу.')
    parser.add_argument('csv')
    parser.add_argument('question')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    args = parser.parse_args()
    result = run_agent(make_client(), args.question, args.csv, model=args.model)
    print(result.answer)
    print(f'[{result.stop_reason}; итераций: {result.iterations}; вызовов инструментов: {result.tool_calls}; '
          f'токены: {result.prompt_tokens}+{result.completion_tokens}]', file=sys.stderr)


if __name__ == '__main__':
    main()
