"""Vérifications sur CPU : python -m unittest discover -s tests -v."""

import io
import unittest

import torch
from torch import nn

from model.video_head import VideoHead


class VideoHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.head = VideoHead().cpu()

    def test_contract_including_single_sequence(self) -> None:
        self.head.eval()
        with torch.no_grad():
            for batch_size in (1, 4):
                with self.subTest(batch_size=batch_size):
                    logits = self.head(torch.rand(batch_size, 16, 128))
                    self.assertEqual(logits.shape, (batch_size,))
                    self.assertTrue(torch.isfinite(logits).all().item())

    def test_training_reaches_all_frames_and_parameters(self) -> None:
        features = torch.rand(2, 16, 128, requires_grad=True)
        labels = torch.tensor([0.0, 1.0])
        optimizer = torch.optim.SGD(self.head.parameters(), lr=0.01)
        before = self.head.classifier.weight.detach().clone()

        loss = nn.BCEWithLogitsLoss()(self.head(features), labels)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all().item())
        self.assertTrue((features.grad.abs().sum(dim=-1) > 0).all().item())
        for name, parameter in self.head.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

        optimizer.step()
        self.assertFalse(torch.equal(before, self.head.classifier.weight))

    def test_checkpoint_preserves_predictions(self) -> None:
        features = torch.rand(2, 16, 128)
        buffer = io.BytesIO()
        torch.save(self.head.state_dict(), buffer)
        buffer.seek(0)
        restored = VideoHead().cpu().eval()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        self.head.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(features), self.head(features))


if __name__ == "__main__":
    unittest.main()
