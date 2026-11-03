import importlib.util
from pathlib import Path
import unittest

import torch


def load_module(name):
    path = Path(__file__).resolve().parents[1] / "network" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select = load_module("attention_targets").select_context_targets
Loss = load_module("loss_tool").Loss


def labels(boxes, batches, classes):
    return {"bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "batch_id": torch.tensor(batches, dtype=torch.long).reshape(-1, 1),
            "cls": torch.tensor(classes, dtype=torch.long)}


class AttentionTargetTests(unittest.TestCase):
    def test_only_context_boxes_preserving_batch_and_geometry(self):
        targets = labels([[.25, .25, .5, .5], [.75, .75, .25, .25],
                          [.75, .75, .125, .125], [.25, .25, .125, .125],
                          [.5, .5, .25, .25]], [0, 0, 0, 0, 1], [4, 1, 2, 3, 4])
        original = {k: v.clone() for k, v in targets.items()}
        result = select(targets)
        for key in targets:
            torch.testing.assert_close(result[key], original[key][[0, 4]])
            torch.testing.assert_close(targets[key], original[key])

    def test_empty_or_no_context(self):
        for targets in (labels([], [], []), labels([[.5, .5, .5, .5]], [0], [1])):
            result = select(targets)
            self.assertEqual(result["bboxes"].shape, (0, 4))
            self.assertEqual(result["batch_id"].shape, (0, 1))

    def test_context_loss_without_any_detection_objects(self):
        # Supervise the context even when there are no detection objects.
        targets = labels([[.25, .25, .5, .5], [.875, .875, .125, .125]], [0, 0], [4, 1])
        logits = torch.zeros(2, 1, 32, 32, requires_grad=True)
        loss = Loss.attention_loss(logits, select(targets))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(logits.grad[0, 0, 8, 8].item(), 0)
        self.assertGreater(logits.grad[0, 0, 28, 28].item(), 0)
        self.assertTrue((logits.grad[1] > 0).all())

    def test_requires_explicit_context_class(self):
        for classes in (None, []):
            with self.assertRaises(ValueError):
                select(labels([], [], []), classes)

    def test_strict_supervises_the_original_boundary_band(self):
        targets = labels([[.5, .5, .4, .4]], [0], [4])
        relaxed_logits = torch.zeros(1, 1, 40, 40, requires_grad=True)
        strict_logits = relaxed_logits.detach().clone().requires_grad_()
        # 关闭比例项，隔离原本被忽略的边界像素。
        Loss.attention_loss(relaxed_logits, targets, lambda_ratio=0).backward()
        Loss.attention_loss_strict(strict_logits, targets, lambda_ratio=0).backward()
        self.assertEqual(relaxed_logits.grad[0, 0, 20, 12].item(), 0)
        self.assertLess(strict_logits.grad[0, 0, 20, 12].item(), 0)

    def test_strict_matches_full_box_loss_and_handles_empty_background(self):
        generator = torch.Generator().manual_seed(42)
        targets = labels([[.5, .5, .4, .4], [.1, .1, .4, .4], [.5, .5, 1, 1]],
                         [0, 0, 2], [4, 4, 4])
        a = torch.randn(3, 1, 40, 40, generator=generator, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        strict = Loss.attention_loss_strict(a, targets)
        reference = Loss.attention_loss(b, targets, shrink_ratio=1, expand_ratio=1)
        torch.testing.assert_close(strict, reference, rtol=1e-6, atol=1e-7)
        strict.backward()
        reference.backward()
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)
        self.assertTrue(torch.isfinite(a.grad).all())
        # 第二幅无场坪图像仍有明确的负监督。
        self.assertTrue((a.grad[1] > 0).all())

    def test_diagnostic_sums_preserve_original_statistics(self):
        module = load_module("attention_targets")
        targets = labels([[.5, .5, .4, .4]], [0], [4])
        logits = torch.zeros(2, 1, 40, 40)
        original = module.attention_region_sums(logits, targets)
        diagnostics, records = module.attention_region_sums(logits, targets,
            include_diagnostics=True, return_records=True)
        self.assertEqual(original, diagnostics[:4])
        self.assertEqual(len(records), 2)
        self.assertIsNone(records[1]["positive_energy_ratio"])
        self.assertAlmostEqual(diagnostics[8] / diagnostics[9], .5)


if __name__ == "__main__":
    unittest.main()
