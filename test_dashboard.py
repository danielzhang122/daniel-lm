import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import dashboard


class DashboardTests(unittest.TestCase):
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
