"""Integration checks for shared architecture, injection location and real detection loss."""
import unittest
import torch
from network import YOLOv8_small, YOLOv8_small_se, YOLOv8_small_cbam, Loss


class ExternalControlsTest(unittest.TestCase):
    def test_shared_weights_injection_and_detection_backward(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        base = YOLOv8_small(nc=1)
        for cls in (YOLOv8_small_se, YOLOv8_small_cbam):
            with self.subTest(model=cls.__name__):
                torch.manual_seed(42)
                model = cls(nc=1)
                for key, value in base.state_dict().items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
                seen = []
                hook = model.external_attention.register_forward_hook(
                    lambda m, inputs, output: seen.append(tuple(output.shape)))
                model.eval()
                with torch.no_grad():
                    output = model(torch.randn(1, 3, 640, 640))
                hook.remove()
                self.assertEqual(seen, [(1, 128, 40, 40)])
                self.assertEqual(output[0].shape, (1, 5, 8000))
                model.train()
                prediction = model(torch.randn(2, 3, 64, 64))
                target = {"batch_id": torch.tensor([0, 1]),
                          "cls": torch.zeros(2, 1),
                          "bboxes": torch.tensor([[.5, .5, .25, .25]] * 2)}
                loss, _ = Loss(model)(prediction, target)
                self.assertTrue(torch.isfinite(loss).all())
                loss.sum().backward()
                for name, param in model.external_attention.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
