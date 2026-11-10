"""Tests that need no API key: python -m unittest discover agent"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csv_agent  # noqa: E402
import evaluate  # noqa: E402


def call(name, arguments, id_='c1'):
    return NS(id=id_, function=NS(name=name, arguments=arguments))


def reply(content=None, calls=None, finish='stop'):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls), finish_reason=finish)],
              usage=NS(prompt_tokens=10, completion_tokens=5))


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv = Path(self.tmp.name) / 'data.csv'
        self.csv.write_text('a,b\n1,2\n3,4\n', encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def test_runs_pandas_code(self):
        self.assertEqual(csv_agent.run_python('print(df.a.sum())', self.csv).strip(), '4')

    def test_api_key_is_hidden(self):
        with mock.patch.dict(os.environ, {'OPENROUTER_API_KEY': 'secret-value'}):
            output = csv_agent.run_python("import os\nprint(os.environ.get('OPENROUTER_API_KEY'))", self.csv)
        self.assertNotIn('secret-value', output)

    def test_error_timeout_and_output_limit(self):
        self.assertIn('ZeroDivisionError', csv_agent.run_python('1/0', self.csv))
        self.assertIn('не завершился', csv_agent.run_python('import time\ntime.sleep(5)', self.csv, timeout=1))
        self.assertIn('обрезан', csv_agent.run_python("print('x' * 100)", self.csv, max_output=10))

    def test_describe(self):
        self.assertEqual(json.loads(csv_agent.describe_csv(self.csv))['columns'], ['a', 'b'])

    def test_agent_loop(self):
        client = FakeClient([
            reply(calls=[call('run_python', json.dumps({'code': 'print(df.b.max())'}))], finish='tool_calls'),
            reply(content='Максимум найден.\nANSWER: 4')])
        result = csv_agent.run_agent(client, 'Каков максимум b?', self.csv, model='m')
        self.assertEqual(evaluate.extract_answer(result.answer), '4')
        self.assertEqual((result.tool_calls, result.iterations, result.prompt_tokens), (1, 2, 20))
        tool_message = next(m for m in client.requests[1]['messages'] if m['role'] == 'tool')
        self.assertEqual((tool_message['role'], tool_message['content'].strip()), ('tool', '4'))

    def test_bad_tool_input_reaches_model(self):
        client = FakeClient([
            reply(calls=[call('run_python', '{bad'), call('other', '{}', 'c2')], finish='tool_calls'),
            reply(content='ANSWER: 1')])
        csv_agent.run_agent(client, 'q', self.csv, model='m')
        outputs = [m['content'] for m in client.requests[1]['messages'] if m['role'] == 'tool']
        self.assertIn('не являются JSON', outputs[0])
        self.assertIn('неизвестный инструмент', outputs[1])

    def test_iteration_limit(self):
        client = FakeClient([reply(calls=[call('describe_csv', '{}')], finish='tool_calls')] * 2)
        result = csv_agent.run_agent(client, 'q', self.csv, model='m', max_iterations=2)
        self.assertEqual((result.stop_reason, result.answer), ('max_iterations', None))

    def test_empty_final_message_gets_one_reminder(self):
        client = FakeClient([reply(content=''), reply(content='ANSWER: 7')])
        result = csv_agent.run_agent(client, 'q', self.csv, model='m')
        self.assertEqual((result.answer, result.iterations), ('ANSWER: 7', 2))
        self.assertIn(csv_agent.ANSWER_REMINDER, [m['content'] for m in result.messages])
        client = FakeClient([reply(content=''), reply(content='')])
        self.assertEqual(csv_agent.run_agent(client, 'q', self.csv, model='m').iterations, 2)

    def test_length_stop_is_reported(self):
        result = csv_agent.run_agent(FakeClient([reply(content='обрезано', finish='length')]), 'q', self.csv, model='m')
        self.assertEqual(result.stop_reason, 'length')

    def test_retry_only_temporary_errors(self):
        temporary = Exception('busy')
        temporary.status_code = 429
        client = FakeClient([temporary, reply(content='ANSWER: 1')])
        with mock.patch('time.sleep'):
            self.assertEqual(csv_agent.run_agent(client, 'q', self.csv, model='m').answer, 'ANSWER: 1')
        permanent = Exception('bad key')
        permanent.status_code = 401
        with self.assertRaises(Exception):
            csv_agent.run_agent(FakeClient([permanent]), 'q', self.csv, model='m')


FRACTION = dict(metric='occupancy_mean', unit='fraction_0_1', absolute_tolerance='1e-06', expected_value='0.4158', month='2026-08')
COUNT = dict(metric='valid_n', unit='rows', absolute_tolerance='0', expected_value='21765', month='2026-05')
TIES = dict(metric='top_parking_all_ties', unit='id', absolute_tolerance='0', expected_value='15;80', month='2026-06')


class Scoring(unittest.TestCase):
    def test_extract(self):
        self.assertEqual(evaluate.extract_answer('текст\n**ANSWER: 12**\n'), '12')
        self.assertEqual(evaluate.extract_answer('ANSWER: 1\nANSWER: 2'), '2')
        self.assertIsNone(evaluate.extract_answer('нет ответа'))

    def test_numbers(self):
        self.assertEqual(evaluate.parse_number('21 765'), (21765.0, False))
        self.assertEqual(evaluate.parse_number('41,58 %'), (41.58, True))
        self.assertIsNone(evaluate.parse_number('около 5'))

    def test_strict_rejects_rounding_and_percent(self):
        precise = dict(FRACTION, expected_value='0.41584')
        self.assertTrue(evaluate.score_strict(precise, '0.41584'))
        self.assertFalse(evaluate.score_strict(precise, '0.4158'))
        self.assertFalse(evaluate.score_strict(precise, '41.584%'))

    def test_protocol_accepts_rounding_and_percent(self):
        precise = dict(FRACTION, expected_value='0.41584')
        for answer in ('0.4158', '41.58%', '41,58', '0.41584'):
            self.assertTrue(evaluate.score_protocol(precise, answer), answer)
        self.assertFalse(evaluate.score_protocol(precise, '0.42'))

    def test_tolerance_covers_low_shares(self):
        self.assertTrue(evaluate.score_protocol(dict(FRACTION, expected_value='0.41585'), '0.4158'))

    def test_counts_and_sets_are_exact(self):
        self.assertTrue(evaluate.score_protocol(COUNT, '21 765'))
        self.assertFalse(evaluate.score_protocol(COUNT, '21766'))
        self.assertTrue(evaluate.score_protocol(TIES, '80; 15'))
        self.assertFalse(evaluate.score_protocol(TIES, '15'))

    def test_p1s_alternative(self):
        alternatives = {('2026-06', 'top_parking_all_ties'): '55'}
        self.assertFalse(evaluate.score_protocol(TIES, '55'))
        self.assertTrue(evaluate.score_with_alternative(TIES, '55', alternatives))

    def test_missing_answer(self):
        self.assertFalse(evaluate.score_strict(COUNT, None))
        self.assertFalse(evaluate.score_protocol(COUNT, None))


class Harness(unittest.TestCase):
    def test_questions_resume_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'p/data').mkdir(parents=True)
            (tmp / 'p/tables').mkdir()
            (tmp / 'p/data/parking_places_hourly_202605.csv').write_text('a\n1\n', encoding='utf-8')
            (tmp / 'p/tables/reference_answers.csv').write_text(
                'month,metric,question_ru,expected_value,unit,absolute_tolerance\n'
                '2026-05,valid_n,Сколько?,1,rows,0\n2026-05,data_rows,Сколько строк?,1,rows,0\n', encoding='utf-8')
            args = NS(package=tmp / 'p', months=None, metrics=None, out=tmp / 'r/runs.jsonl', mode='agent', condition='protocol',
                      model='m', limit=1, dry_run=False, sample_rows=5)
            self.assertEqual(evaluate.run_questions(args, FakeClient([reply(content='ANSWER: 1')])), 1)
            args.limit = None
            self.assertEqual(evaluate.run_questions(args, FakeClient([reply(content='ANSWER: 2')])), 1)
            row = evaluate.summarize(args.out)[0]
            self.assertEqual((row['runs'], row['strict'], row['protocol']), (2, 0.5, 0.5))

    def test_package_prompt_includes_rule(self):
        reference = dict(question_ru='Сколько?', processing_rule='P1: rule')
        self.assertIn('P1: rule', evaluate.build_prompt(reference, 'package'))

    def test_failed_run_is_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'runs.jsonl'
            out.write_text(json.dumps(dict(task='questions', mode='agent', model='m', condition='c', month='x', metric='y', error='boom')) + '\n', encoding='utf-8')
            self.assertEqual(evaluate.load_done(out), set())

    def test_llm_prompt_is_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'd.csv'
            path.write_text('\na\n1\n2\n3\n', encoding='utf-8')
            text = evaluate.llm_prompt('Вопрос', path, 2)
            self.assertIn('3 строк данных', text)
            self.assertNotIn('\n3\n', text.split('Инструментов')[1])


if __name__ == '__main__':
    unittest.main()
