import hashlib
from pathlib import Path
import tempfile
import unittest
import torch
from model import GPT
from inspect_model import inspect
from test_tools import tokenizer


class InspectTests(unittest.TestCase):
    def test_reports_real_dimensions_parameters_and_token_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            tok=tokenizer();tok.save(str(root/'tokenizer.json'))
            model=GPT(vocab_size=tok.get_vocab_size(),context=16,width=32,layers=2,heads=4)
            torch.save(dict(config=model.config,model=model.state_dict(),step=7,
                            data={'tokenizer_sha256':hashlib.sha256((root/'tokenizer.json').read_bytes()).hexdigest()}),root/'model.pt')
            data=inspect(root/'model.pt',root/'tokenizer.json','Hello world.')
            self.assertEqual(data['config'],model.config)
            self.assertEqual(data['step'],7)
            self.assertEqual(data['total_parameters'],sum(p.numel() for p in model.parameters()))
            self.assertEqual(len(data['block_parameters']),2)
            self.assertEqual(data['weight_samples'][0]['expand'][0][0],model.blocks[0].mlp[0].weight[0,0].item())
            self.assertEqual(data['weight_samples'][1]['project'][0][0],model.blocks[1].mlp[2].weight[0,0].item())
            self.assertEqual(len(data['weight_samples'][0]['expand']),48)
            self.assertEqual(len(data['weight_samples'][0]['expand'][0]),32)
            self.assertEqual([t['id'] for t in data['tokens']],tok.encode('Hello world.').ids)
            self.assertEqual(data['total_parameters'],data['token_embedding_parameters']+data['position_embedding_parameters']+data['final_norm_parameters']+sum(data['block_parameters']))
            with self.assertRaisesRegex(ValueError,'characters'):
                inspect(root/'model.pt',root/'tokenizer.json',' ')
            (root/'tokenizer.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Tokenizer'):
                inspect(root/'model.pt',root/'tokenizer.json','Hello')


if __name__=='__main__': unittest.main()
