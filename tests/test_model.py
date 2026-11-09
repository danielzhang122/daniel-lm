import unittest
import torch
from model import GPT


class ModelTests(unittest.TestCase):
    def test_future_tokens_cannot_change_past_logits(self):
        torch.manual_seed(0)
        model = GPT(vocab_size=32, context=16, width=32, layers=2, heads=4).eval()
        x = torch.randint(0, 32, (2, 16))
        changed = x.clone()
        changed[:, 8:] = (changed[:, 8:] + 1) % 32
        with torch.no_grad():
            torch.testing.assert_close(model(x)[:, :8], model(changed)[:, :8])

    def test_can_learn_repeated_sequence(self):
        torch.manual_seed(0)
        model = GPT(vocab_size=16, context=8, width=32, layers=1, heads=4)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        x = torch.arange(8).repeat(2, 1)
        target = x + 1
        values = []
        for _ in range(20):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 16), target.reshape(-1))
            values.append(loss.item())
            loss.backward()
            opt.step()
        self.assertLess(values[-1], values[0] / 2)


if __name__ == '__main__':
    unittest.main()
