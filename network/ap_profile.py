"""Descriptive IoU-threshold analysis using the existing AP integration unchanged."""
from .train_base import compute_map_from_records

IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def summarize_ap_profile(metrics):
    """Split the existing ten-threshold mean into two equally weighted bands.

    Band differences describe contributions to AP changes, not causal error types.
    """
    thresholds = metrics['iou_thresholds']
    values = metrics['map_per_iou']
    if len(values) != 10 or len(thresholds) != 10 or any(abs(a-b) > 1e-8 for a,b in zip(thresholds, IOU_THRESHOLDS)):
        raise ValueError('Expected the full IoU 0.50:0.05:0.95 sequence')
    return {
        'ap_integration': 'precision_envelope_recall_change_point_sum',
        'matching': 'existing_project_matching_unchanged',
        'AP50': values[0], 'AP75': values[5], 'AP90': values[8],
        'AP50_95': sum(values)/10,
        'AP50_70': sum(values[:5])/5,
        'AP75_95': sum(values[5:])/5,
        'iou_thresholds': list(thresholds), 'AP_per_iou': list(values),
        'gt_count_per_class': metrics['gt_count_per_class'],
        'valid_classes': metrics['valid_classes'],
        'ap_per_class_by_iou': metrics['ap_per_class_by_iou'],
    }


def profile_records(pred_records, gt_records, num_classes):
    return summarize_ap_profile(compute_map_from_records(
        pred_records, gt_records, num_classes, iou_thresholds=IOU_THRESHOLDS))
