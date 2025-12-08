import os
import torch
import numpy as np
import logging
import json
import shutil
from collections import OrderedDict

from PIL import Image
from glob import glob
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from .evaluator import DatasetEvaluator


class GaemiEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): the name of the dataset.
                It must have the following metadata associated with it:
                "thing_classes", "gt_dir".
        """
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)
        self._working_dir = os.path.join(os.getcwd(), "gaemi_eval")

        # Only the main process handles directory setup
        if comm.is_main_process():
            if os.path.exists(self._working_dir):
                self._logger.info("Cleaning up old prediction files from previous run...")
                shutil.rmtree(self._working_dir)
            os.makedirs(self._working_dir, exist_ok=True)
            self._logger.info(f"Prediction directory ready: {self._working_dir}")

        # Wait for all processes to finish directory creation
        comm.synchronize()


class GaemiSemsegEvaluator(GaemiEvaluator):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)

        # Bring class names from metadata
        self._class_names = getattr(self._metadata, 'all_classes', None)
        if self._class_names is None:
            raise ValueError("Metadata must contain 'all_classes' for GaemiSemsegEvaluator.")

        self._num_classes = len(self._class_names)
        self._ignore_label = getattr(self._metadata, 'ignore_label', 255)

        # Initialize ID mapping: dataset_id -> contiguous_id
        self._dataset_to_contiguous = getattr(
            self._metadata,
            'all_dataset_id_to_contiguous_id',
            None
        )

        if self._dataset_to_contiguous is None:
            raise ValueError("Metadata must contain 'all_dataset_id_to_contiguous_id' for GaemiSemsegEvaluator.")

        # Inverse mapping: contiguous_id -> dataset_id
        self._contiguous_id_to_dataset_id = {v: k for k, v in self._dataset_to_contiguous.items()}

        # Initialize Confusion Matrix
        self.reset()

    def reset(self):
        """Initialize Confusion Matrix before evaluation starts"""
        self._conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1),
            dtype=np.int64
        )

    def _convert_to_contiguous(self, id_map):
        """
        Convert dataset_id map to contiguous_id map.
        Args:
            id_map: [H, W] numpy array with dataset_ids
        Returns:
            [H, W] numpy array with contiguous_ids (ignore = num_classes)
        """
        contiguous_map = np.full(id_map.shape, self._num_classes, dtype=np.int32)  # ignore = num_classes

        for dataset_id, contiguous_id in self._dataset_to_contiguous.items():
            contiguous_map[id_map == dataset_id] = contiguous_id

        return contiguous_map

    def _update_confusion_matrix(self, gt_contiguous, pred_contiguous):
        """
        Update confusion matrix with pixel-level comparison.
        Args:
            gt_contiguous: [H, W] numpy array with GT contiguous_ids
            pred_contiguous: [H, W] numpy array with predicted contiguous_ids
        """
        # Select only valid pixels (excluding ignore label)
        valid_mask = (gt_contiguous < self._num_classes) & (pred_contiguous < self._num_classes)

        gt_valid = gt_contiguous[valid_mask]
        pred_valid = pred_contiguous[valid_mask]

        if len(gt_valid) == 0:
            return

        # Update confusion matrix
        n = self._num_classes + 1  # +1 for ignore
        hist = np.bincount(
            n * gt_valid + pred_valid,
            minlength=n * n
        ).reshape(n, n)

        self._conf_matrix += hist

    def _compute_metrics_from_confusion_matrix(self, conf_matrix):
        # ===== Compute IoU =====
        # Extract metrics from Confusion Matrix
        # Confusion Matrix: row=GT, column=prediction
        tp = conf_matrix.diagonal()[:-1].astype(float)  # True Positives (excluding ignore)
        pos_gt = np.sum(conf_matrix[:-1, :-1], axis=1).astype(float)  # GT pixel count (row sum)
        pos_pred = np.sum(conf_matrix[:-1, :-1], axis=0).astype(float)  # Prediction pixel count (column sum)

        # Compute Accuracy
        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)

        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]

        # Compute IoU
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]

        # ===== Compute mean metrics =====
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid) if np.sum(acc_valid) > 0 else 0
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid) if np.sum(iou_valid) > 0 else 0
        pacc = np.sum(tp) / np.sum(pos_gt) if np.sum(pos_gt) > 0 else 0

        # Frequency Weighted IoU
        class_weights = pos_gt / np.sum(pos_gt) if np.sum(pos_gt) > 0 else np.zeros(self._num_classes)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid]) if np.sum(iou_valid) > 0 else 0

        # ===== Construct results =====
        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc

        # Per-class IoU and Accuracy
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i] if not np.isnan(iou[i]) else 0.0
            res[f"ACC-{name}"] = 100 * acc[i] if not np.isnan(acc[i]) else 0.0

        ordereddict_res = OrderedDict([('sem_seg', res)])

        return ordereddict_res, iou

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            if 'sem_seg' not in output:
                raise ValueError("Expected 'sem_seg' in model output for semantic segmentation.")

            # Model output: contiguous_id
            # Note: Mask2Former uses 0 as background, so actual classes start from 1
            pred_contiguous_raw = output['sem_seg'].argmax(dim=0).to(self._cpu_device).numpy()

            # Initialize with 255 (ignore label, prevents black color)
            pred_dataset = np.full(pred_contiguous_raw.shape, 255, dtype=np.uint8)

            # Check if identity mapping (1:1, 2:2, ...)
            is_identity_mapping = all(
                k == v for k, v in self._contiguous_id_to_dataset_id.items()
            )
            if is_identity_mapping:
                # Identity mapping: use model output directly (0=background -> 255)
                pred_dataset = pred_contiguous_raw.copy().astype(np.uint8)
                pred_dataset[pred_contiguous_raw == 0] = 255  # background -> ignore
            else:
                # Non-identity mapping: need to convert contiguous_id → dataset_id
                # Convert model output to 0-based (subtract 1)
                pred_contiguous = pred_contiguous_raw.copy()
                pred_contiguous[pred_contiguous_raw > 0] = pred_contiguous_raw[pred_contiguous_raw > 0] - 1

                for contiguous_id, dataset_id in self._contiguous_id_to_dataset_id.items():
                    if dataset_id > 255:
                        self._logger.warning(f"dataset_id {dataset_id} exceeds uint8 range, clipping to 255")
                        dataset_id = 255
                    pred_dataset[pred_contiguous == contiguous_id] = dataset_id

            file_name = input.get('image_id', 'unknown')
            pred_file = os.path.join(self._working_dir, f"{file_name}.png")
            Image.fromarray(pred_dataset).save(pred_file)

    def evaluate(self):
        gt_json_path = getattr(self._metadata, 'gt_json_path', None)

        if gt_json_path is None:
            raise ValueError("Metadata must contain 'gt_json_path' for GaemiSemsegEvaluator.")

        if not os.path.exists(gt_json_path):
            self._logger.error(f"GT JSON file not found: {gt_json_path}")
            raise ValueError(f"GT JSON file not found: {gt_json_path}")

        # Get mount_path from metadata for path compatibility
        mount_path = getattr(self._metadata, 'mount_path', '')

        with open(gt_json_path, 'r') as f:
            gt_data = json.load(f)

        pred_files = glob(os.path.join(self._working_dir, "*.png"))

        # Map image_id → prediction path
        pred_dict = {}
        for pred_path in pred_files:
            base_name = os.path.splitext(os.path.basename(pred_path))[0]
            pred_dict[base_name] = pred_path

        processed_count = 0
        skipped_count = 0

        for gt_item in gt_data:
            image_id = gt_item.get('image_id', 'unknown')

            # Find prediction file
            if image_id not in pred_dict:
                self._logger.warning(f"No prediction found for image_id: {image_id}")
                skipped_count += 1
                continue

            # Load GT PNG
            gt_seg_path = gt_item.get('sem_seg_file_name')
            # Convert to absolute path if mount_path is provided and path is relative
            if mount_path and not os.path.isabs(gt_seg_path):
                full_gt_seg_path = os.path.join(mount_path, gt_seg_path)
            else:
                full_gt_seg_path = gt_seg_path

            if not full_gt_seg_path or not os.path.exists(full_gt_seg_path):
                self._logger.warning(f"GT file not found: {full_gt_seg_path}")
                skipped_count += 1
                continue

            gt_png = np.array(Image.open(full_gt_seg_path))

            # Load prediction PNG (saved as dataset_id in process())
            pred_png = np.array(Image.open(pred_dict[image_id]))

            # Check size
            if gt_png.shape[:2] != pred_png.shape[:2]:
                self._logger.warning(
                    f"Shape mismatch for {image_id}: GT {gt_png.shape} vs Pred {pred_png.shape}"
                )
                skipped_count += 1
                continue

            # Convert to grayscale (if RGB)
            if len(gt_png.shape) == 3:
                gt_png = gt_png[:, :, 0] if gt_png.shape[2] >= 1 else gt_png
            if len(pred_png.shape) == 3:
                pred_png = pred_png[:, :, 0] if pred_png.shape[2] >= 1 else pred_png

            gt_contiguous = self._convert_to_contiguous(gt_png)
            pred_contiguous = self._convert_to_contiguous(pred_png)

            self._update_confusion_matrix(gt_contiguous, pred_contiguous)
            processed_count += 1

        self._logger.info(f"Processed {processed_count} image pairs")
        if skipped_count > 0:
            self._logger.warning(f"Skipped {skipped_count} images due to errors")

        comm.synchronize()

        # Gather confusion matrices from all GPUs
        all_conf_matrices = comm.gather(self._conf_matrix, dst=0)

        if not comm.is_main_process():
            return OrderedDict({"sem_seg": {}})

        # Main process: sum confusion matrices from all GPUs
        total_conf_matrix = sum(all_conf_matrices)

        self._logger.info("Starting semantic segmentation evaluation...")
        res, iou = self._compute_metrics_from_confusion_matrix(total_conf_matrix)

        self._logger.info("=" * 70)
        self._logger.info("Semantic Segmentation Evaluation Results:")
        self._logger.info(f"  mIoU:  {res['sem_seg']['mIoU']:.2f}")
        self._logger.info(f"  fwIoU: {res['sem_seg']['fwIoU']:.2f}")
        self._logger.info(f"  mACC:  {res['sem_seg']['mACC']:.2f}")
        self._logger.info(f"  pACC:  {res['sem_seg']['pACC']:.2f}")
        self._logger.info("-" * 70)
        self._logger.info("Per-class IoU:")
        for i, name in enumerate(self._class_names):
            if not np.isnan(iou[i]):
                self._logger.info(f"  {name:20s}: {100 * iou[i]:5.2f}")
        self._logger.info("=" * 70)

        self._logger.info(f"Final evaluation result: {res}")
        return res


class GaemiPanopticEvaluator(GaemiEvaluator):
    """
    Should be implemented later.
    """

    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        raise NotImplementedError("GaemiPanopticEvaluator is not implemented yet.")
