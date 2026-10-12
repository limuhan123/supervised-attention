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


if __name__ == "__main__":
    unittest.main()
