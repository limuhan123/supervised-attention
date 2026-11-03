import unittest
import torch
from network.train_base import compute_ap, compute_map_from_records
from network.ap_profile import profile_records, summarize_ap_profile


class APProfileTests(unittest.TestCase):
    def test_change_point_integral_is_not_101_point_sampling(self):
        # Only half the objects recovered, with perfect precision: integral = 0.5.
        self.assertAlmostEqual(compute_ap(torch.tensor([.5]), torch.tensor([1.])), .5)

    def test_precision_envelope_integral(self):
        recall = torch.tensor([0., .5, .5, 1.])
        precision = torch.tensor([0., .5, 1./3., .5])
        self.assertAlmostEqual(compute_ap(recall, precision), .5)

    def test_profile_preserves_legacy_metric_and_high_iou_drop(self):
        box = torch.tensor([0., 0., 1., 1.])
        gt = [{'img_id': 0, 'cls': 0, 'box': box}]
        pred = [{'img_id': 0, 'cls': 0, 'conf': .9,
                 'box': torch.tensor([0., 0., .8, .8])}]
        profile = profile_records(pred, gt, 1)
        metrics = compute_map_from_records(pred, gt, 1, profile['iou_thresholds'])
        self.assertEqual(profile['AP50_95'], metrics['map50_95'])
        self.assertEqual(profile['AP50'], 1.)
        self.assertEqual(profile['AP75'], 0.)
        self.assertAlmostEqual(profile['AP50_95'],
                               .5*(profile['AP50_70']+profile['AP75_95']))

    def test_no_predictions_and_threshold_validation(self):
        gt = [{'img_id': 0, 'cls': 0, 'box': torch.tensor([0., 0., 1., 1.])}]
        self.assertEqual(profile_records([], gt, 1)['AP50_95'], 0.)
        with self.assertRaises(ValueError):
            summarize_ap_profile({'iou_thresholds': [.5], 'map_per_iou': [1.]})


if __name__ == '__main__':
    unittest.main()
