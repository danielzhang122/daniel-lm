import unittest
import numpy as np
from cleaning import clean_article
from training_utils import (BlockSampler, learning_rate, validate_transfer,
                            validate_resume_training, optimizer_state_with_backend,
                            resolve_model_shape)


class TrainingTests(unittest.TestCase):
    def test_optimizer_backend_preserves_state_without_mutating_checkpoint(self):
        state={'state': {1: {'step': 42, 'exp_avg': 'existing'}},
               'param_groups': [{'params':[1], 'fused':None, 'foreach':None, 'lr':0.001}]}
        fused=optimizer_state_with_backend(state,'fused')
        self.assertTrue(fused['param_groups'][0]['fused'])
        self.assertFalse(fused['param_groups'][0]['foreach'])
        self.assertEqual(fused['state'][1]['step'],42)
        self.assertIsNone(state['param_groups'][0]['fused'])
        default=optimizer_state_with_backend(fused,'default')
        self.assertIsNone(default['param_groups'][0]['fused'])
        self.assertIsNone(default['param_groups'][0]['foreach'])
        self.assertIs(optimizer_state_with_backend(state,None),state)
        with self.assertRaises(ValueError): optimizer_state_with_backend(state,'invalid')

    def test_model_shape_defaults_overrides_and_saved_checkpoint(self):
        none = dict(context=None, width=None, layers=None, heads=None)
        self.assertEqual(resolve_model_shape(none), dict(context=512, width=512, layers=8, heads=8))
        self.assertEqual(resolve_model_shape({**none, 'width': 768, 'heads': 12})['width'], 768)
        saved = dict(vocab_size=16384, context=1024, width=768, layers=12, heads=12)
        resumed = resolve_model_shape(none, saved)
        self.assertEqual(resumed, dict(context=1024, width=768, layers=12, heads=12))
        self.assertEqual(resolve_model_shape({**none, 'layers': 12}, saved), resumed)
        with self.assertRaisesRegex(ValueError, 'does not match the saved model'):
            resolve_model_shape({**none, 'width': 512}, saved)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            resolve_model_shape({**none, 'width': 100, 'heads': 12})
        with self.assertRaisesRegex(ValueError, 'positive'):
            resolve_model_shape({**none, 'layers': 0})

    def test_resume_accepts_only_same_effective_batch_and_schedule(self):
        previous=dict(sampler='blocks-v1',batch=2,accum=8,horizon=10000,lr=.001)
        next_settings={**previous,'batch':16,'accum':1}
        validate_resume_training(previous,next_settings)
        with self.assertRaisesRegex(ValueError,'effective batch'):
            validate_resume_training(previous,{**next_settings,'accum':2})
        with self.assertRaisesRegex(ValueError,'schedule'):
            validate_resume_training(previous,{**next_settings,'horizon':20000})
        with self.assertRaises(ValueError):
            validate_resume_training({**previous,'sampler':'other'}, {**next_settings,'sampler':'other'})

    def test_regrouped_batches_preserve_block_order(self):
        small=BlockSampler(101,4,position=16)
        large=BlockSampler(101,4,position=16)
        np.testing.assert_array_equal(np.concatenate([small.next_indices(2) for _ in range(8)]),large.next_indices(16))

    def test_regrouped_batches_produce_equivalent_gradients(self):
        import copy
        import torch
        from model import GPT
        torch.manual_seed(42)
        model=GPT(vocab_size=32,context=8,width=16,layers=1,heads=4)
        reference=copy.deepcopy(model)
        tokens=torch.randint(0,32,(16,9))
        for batch in tokens.split(2):
            logits=reference(batch[:,:-1])
            (torch.nn.functional.cross_entropy(logits.reshape(-1,32),batch[:,1:].reshape(-1))/8).backward()
        logits=model(tokens[:,:-1])
        torch.nn.functional.cross_entropy(logits.reshape(-1,32),tokens[:,1:].reshape(-1)).backward()
        for p,q in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6)

    def test_transfer_accepts_new_data_but_rejects_wrong_tokenizer_or_model(self):
        config = {'width': 512, 'vocab_size': 16384}
        source = {'config': config, 'data': {'tokenizer_sha256': 'abc', 'tokens': 100}}
        validate_transfer(source, config, {'tokenizer_sha256': 'abc', 'tokens': 1000})
        with self.assertRaisesRegex(ValueError, 'identical tokenizer'):
            validate_transfer(source, config, {'tokenizer_sha256': 'different'})
        with self.assertRaisesRegex(ValueError, 'architecture'):
            validate_transfer(source, {'width': 768}, {'tokenizer_sha256': 'abc'})

    def test_cleanup(self):
        self.assertEqual(clean_article('A sources discussion.\n\nHistory\nBody\n\nReferences\nCitation\nLiving people'),
                         'A sources discussion.\n\nHistory\nBody')
        self.assertEqual(clean_article('Body\n== External links ==\nURL'), 'Body')
        self.assertEqual(clean_article('Body\nCategory:People\nMore prose'), 'Body\nMore prose')

    def test_epoch_coverage_and_resume(self):
        sampler = BlockSampler(23, 4)
        first = sampler.next_indices(6)
        self.assertEqual(set(first[:, :-1].flatten()), set(range(23)))
        self.assertEqual(len(set(first[:, 0])), 6)
        resumed = BlockSampler(23, 4, position=6)
        np.testing.assert_array_equal(sampler.next_indices(9), resumed.next_indices(9))

    def test_schedule(self):
        self.assertAlmostEqual(learning_rate(1, 1, .1, 10, 100), .1)
        self.assertEqual(learning_rate(10, 1, .1, 10, 100), 1)
        self.assertAlmostEqual(learning_rate(100, 1, .1, 10, 100), .1)
        self.assertAlmostEqual(learning_rate(200, 1, .1, 10, 100), .1)
        values = [learning_rate(i, 1, .1, 10, 100) for i in range(10, 101)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))


if __name__ == '__main__':
    unittest.main()
