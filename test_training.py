import unittest
import numpy as np
from cleaning import clean_article
from training_utils import BlockSampler, learning_rate, validate_transfer


class TrainingTests(unittest.TestCase):
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
