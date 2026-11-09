import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import dashboard


class DashboardTests(unittest.TestCase):
    def test_generation_rejects_invalid_requests(self):
        for payload in [[], {'run': '../full'}, {'run': 'full', 'prompt': ''},
                        {'run': 'full', 'prompt': 'Hi', 'tokens': 5000},
                        {'run': 'full', 'prompt': 'Hi', 'temperature': float('nan')}]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                dashboard.generation_command(payload)

    def test_generation_uses_cpu_and_preserves_prompt_as_single_argument(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(dashboard, 'ROOT', Path(temp)):
            folder = Path(temp) / 'runs' / 'full'
            folder.mkdir(parents=True)
            (folder / 'latest.pt').touch()
            (folder / 'run.json').write_text(json.dumps({'data': 'E:/wiki-llm/data/full'}))
            prompt = 'Text; $(not a command) <script>test</script>'
            command = dashboard.generation_command({'run': 'full', 'prompt': prompt})
            self.assertEqual(command[command.index('--device')+1], 'cpu')
            self.assertEqual(command[command.index('--prompt')+1], prompt)
            self.assertEqual(command[command.index('--tokenizer')+1].replace('\\', '/'),
                             'E:/wiki-llm/data/full/tokenizer.json')

    def test_empty_workspace(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(dashboard, 'ROOT', Path(temp)):
            self.assertEqual(dashboard.snapshot()['runs'], [])

    def test_partial_log_and_matching_dataset(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(dashboard, 'ROOT', Path(temp)):
            root = Path(temp)
            run = root / 'runs' / 'example'
            data = root / 'data' / 'example'
            run.mkdir(parents=True)
            data.mkdir(parents=True)
            (run / 'metrics.jsonl').write_text('{"step": 10, "train_loss": 4.2}\n{"step":', encoding='utf-8')
            (data / 'metadata.json').write_text(json.dumps({'tokens': {'train': 123}}))
            result = dashboard.snapshot()['runs'][0]
            self.assertEqual(len(result['rows']), 1)
            self.assertEqual(result['dataset']['tokens']['train'], 123)
            self.assertFalse(result['checkpoint'])


if __name__ == '__main__':
    unittest.main()
