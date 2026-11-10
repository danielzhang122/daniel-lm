import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import dashboard
from dialogue import rag_user_message


class DashboardTests(unittest.TestCase):
    def test_compact_metrics_preserves_first_latest_and_last_duplicate(self):
        rows = [dict(step=i, train_loss=float(i), tokens_per_second=i) for i in range(100)]
        rows += [dict(step=i, val_loss=float(i)) for i in range(0,100,10)]
        rows += [dict(step=99, train_loss=123.0, tokens_per_second=456)]
        compact = dashboard.compact_metrics(rows, train_limit=7, validation_limit=4)
        train = [r for r in compact if 'train_loss' in r]
        validation = [r for r in compact if 'val_loss' in r]
        self.assertEqual(len(train),7)
        self.assertEqual(len(validation),4)
        self.assertEqual([train[0]['step'],train[-1]['step']],[0,99])
        self.assertEqual(train[-1]['tokens_per_second'],456)
        self.assertEqual([validation[0]['step'],validation[-1]['step']],[0,90])

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

    def test_retrieval_adds_passages_to_the_last_message_only(self):
        found=[dict(id=7,title='France',url='u',text='Paris is the capital of France.')]
        history=[{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello.'},{'role':'user','content':'What is the capital of France?'}]
        with tempfile.TemporaryDirectory() as temp, patch.object(dashboard, 'ROOT', Path(temp)):
            folder=Path(temp)/'runs'/'chat';folder.mkdir(parents=True);(folder/'latest.pt').touch()
            (folder/'run.json').write_text(json.dumps({'data':'E:/wiki-llm/data/full','phase':'dialogue'}))
            def sent(command): return json.loads(command[command.index('--messages-json')+1])
            with patch.object(dashboard.library,'passages',return_value=found):
                command,sources=dashboard.generation_request({'run':'chat','messages':history,'retrieve':True})
            self.assertEqual(sent(command)[:2],history[:2])
            self.assertEqual(sent(command)[-1]['content'],rag_user_message(found,'What is the capital of France?'))
            self.assertEqual(sources[0]['title'],'France')
            with patch.object(dashboard.library,'passages',return_value=[]):
                command,sources=dashboard.generation_request({'run':'chat','messages':history,'retrieve':True})
            self.assertEqual((sent(command),sources),(history,[]))
            with patch.object(dashboard.library,'passages',side_effect=AssertionError('not called')):
                command,sources=dashboard.generation_request({'run':'chat','messages':history})
            self.assertEqual((sent(command),sources),(history,[]))

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
