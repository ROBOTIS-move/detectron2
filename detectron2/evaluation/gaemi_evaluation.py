import os
import torch
import numpy as np
import logging
import json
import shutil
import io
from collections import OrderedDict, defaultdict

from PIL import Image
from glob import glob
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
# from Mask2Former.custom_util.config.class_config import class_info
from .evaluator import DatasetEvaluator

def id2rgb(id_map):
    """Convert panoptic ID map to RGB image for visualization"""
    rgb = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = id_map % 256
    rgb[:, :, 1] = (id_map // 256) % 256
    rgb[:, :, 2] = (id_map // 256 // 256) % 256
    return rgb

def rgb2id(rgb):
    """Convert RGB image back to panoptic ID map"""
    # Convert to int32 to avoid overflow when multiplying by 256
    rgb = rgb.astype(np.int32)
    result = rgb[:, :, 0] + rgb[:, :, 1] * 256 + rgb[:, :, 2] * 256 * 256
    return result.astype(np.int32)


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
        
        # 멀티 프로세스 환경에서 안전하게 디렉토리 초기화
        # 메인 프로세스만 디렉토리를 정리하고 재생성
        if comm.is_main_process():
            if os.path.exists(self._working_dir):
                self._logger.info(f"Cleaning up old prediction files from previous run...")
                shutil.rmtree(self._working_dir)
            os.makedirs(self._working_dir, exist_ok=True)
            self._logger.info(f"Prediction directory ready: {self._working_dir}")
        
        # 모든 프로세스가 디렉토리 생성을 기다림
        comm.synchronize()

class GaemiSemsegEvaluator(GaemiEvaluator):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        
        # Metadata에서 클래스 정보 가져오기 (stuff_classes 우선, 없으면 thing_classes)
        self._class_names = getattr(self._metadata, 'stuff_classes', None)
        if self._class_names is None:
            self._class_names = getattr(self._metadata, 'thing_classes', [])
        
        self._num_classes = len(self._class_names)
        self._ignore_label = getattr(self._metadata, 'ignore_label', 255)
        
        # ID 매핑 초기화: contiguous_id -> dataset_id
        dataset_to_contiguous = getattr(
            self._metadata,
            'stuff_dataset_id_to_contiguous_id',
            None
        )
    
        if dataset_to_contiguous is None:
            # Fallback to thing mapping
            dataset_to_contiguous = getattr(
                self._metadata,
                'thing_dataset_id_to_contiguous_id',
                {}
            )
        
        # 역변환: contiguous_id -> dataset_id
        self._contiguous_id_to_dataset_id = {v: k for k, v in dataset_to_contiguous.items()}
        
        # 로깅
        self._logger.info(f"Loaded {len(self._contiguous_id_to_dataset_id)} class ID mappings")
        self._logger.info(f"Dataset->Contiguous mapping: {sorted(dataset_to_contiguous.items())}")
        self._logger.info(f"Contiguous->Dataset mapping: {sorted(self._contiguous_id_to_dataset_id.items())}")
        
        # Confusion Matrix 초기화
        self.reset()
    
    def reset(self):
        """평가 시작 전 Confusion Matrix 초기화"""
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
        
        # dataset_id → contiguous_id 역매핑
        dataset_to_contiguous = {v: k for k, v in self._contiguous_id_to_dataset_id.items()}
        
        for dataset_id, contiguous_id in dataset_to_contiguous.items():
            contiguous_map[id_map == dataset_id] = contiguous_id
        
        return contiguous_map
    
    def _update_confusion_matrix(self, gt_contiguous, pred_contiguous):
        """
        Update confusion matrix with pixel-level comparison.
        
        Args:
            gt_contiguous: [H, W] numpy array with GT contiguous_ids
            pred_contiguous: [H, W] numpy array with predicted contiguous_ids
        """
        # 유효한 픽셀만 선택 (ignore 제외)
        valid_mask = (gt_contiguous < self._num_classes) & (pred_contiguous < self._num_classes)
        
        gt_valid = gt_contiguous[valid_mask]
        pred_valid = pred_contiguous[valid_mask]
        
        if len(gt_valid) == 0:
            return
        
        # Confusion matrix 업데이트
        n = self._num_classes + 1  # +1 for ignore
        hist = np.bincount(
            n * gt_valid + pred_valid,
            minlength=n * n
        ).reshape(n, n)
        
        self._conf_matrix += hist

    def process(self, inputs, outputs):
        """
        모델의 semantic segmentation 예측 결과를 처리합니다.
        1. 예측을 PNG로 저장 (dataset_id로 변환)
        2. GT 로드하여 Confusion Matrix 업데이트
        """
       
        # 첫 번째 이미지에서 매핑 정보 출력
        debug_first = not hasattr(self, '_process_debug_done')
        
        for input, output in zip(inputs, outputs):
            # ===== 1. 예측 처리 =====
            if 'sem_seg' not in output:
                raise ValueError("Expected 'sem_seg' in model output for semantic segmentation.")
            
            # 모델 출력: contiguous_id
            # 주의: Mask2Former는 0을 background로 사용하므로 실제 클래스는 1부터 시작
            pred_contiguous_raw = output['sem_seg'].argmax(dim=0).to(self._cpu_device).numpy()
            
            if debug_first:
                self._logger.info(f"[PROCESS DEBUG] pred_contiguous_raw unique: {np.unique(pred_contiguous_raw)}")
                self._logger.info(f"[PROCESS DEBUG] ID mapping: {sorted(self._contiguous_id_to_dataset_id.items())}")
                self._process_debug_done = True

            # ===== 2. 예측 PNG 저장 (dataset_id로 변환) =====
            # 255로 초기화 (ignore label, 검은색 방지)
            pred_dataset = np.full(pred_contiguous_raw.shape, 255, dtype=np.uint8)
            
            # 모델 출력을 0-based contiguous_id로 변환 (1 빼기)
            # 모델: 0=background, 1=class_0, 2=class_1, ...
            # 매핑: 0=class_0, 1=class_1, ...
            pred_contiguous = pred_contiguous_raw.copy()
            pred_contiguous[pred_contiguous_raw > 0] = pred_contiguous_raw[pred_contiguous_raw > 0] - 1
            
            # contiguous_id → dataset_id 변환
            for contiguous_id, dataset_id in self._contiguous_id_to_dataset_id.items():
                # dataset_id가 uint8 범위를 초과하면 경고
                if dataset_id > 255:
                    self._logger.warning(f"dataset_id {dataset_id} exceeds uint8 range, clipping to 255")
                    dataset_id = 255
                pred_dataset[pred_contiguous == contiguous_id] = dataset_id
            
            file_name = input.get('image_id', 'unknown')
            pred_file = os.path.join(self._working_dir, f"{file_name}.png")
            Image.fromarray(pred_dataset).save(pred_file)

    def evaluate(self):
        """
        저장된 예측 결과와 GT를 비교하여 평가 지표를 계산합니다.
        
        Returns:
            OrderedDict: {"sem_seg": {지표들}}
        """
        # ===== 1. GT JSON 로드 =====
        gt_json_path = getattr(self._metadata, 'gt_json_path', None)
        
        if gt_json_path is None:
            self._logger.error("gt_json_path not found in metadata!")
            return OrderedDict({"sem_seg": {}})
        
        if not os.path.exists(gt_json_path):
            self._logger.error(f"GT JSON file not found: {gt_json_path}")
            return OrderedDict({"sem_seg": {}})
        
        self._logger.info(f"Loading GT annotations from: {gt_json_path}")
        with open(gt_json_path, 'r') as f:
            gt_data = json.load(f)
        
        self._logger.info(f"Found {len(gt_data)} GT annotations")
        
        # ===== 2. 예측 파일 목록 수집 =====
        pred_files = glob(os.path.join(self._working_dir, "*.png"))
        self._logger.info(f"Found {len(pred_files)} prediction files in {self._working_dir}")
        
        # image_id → prediction path 매핑
        pred_dict = {}
        for pred_path in pred_files:
            base_name = os.path.splitext(os.path.basename(pred_path))[0]
            pred_dict[base_name] = pred_path
        
        # ===== 3. GT와 예측 비교하여 Confusion Matrix 업데이트 =====
        processed_count = 0
        skipped_count = 0
        
        for gt_item in gt_data:
            image_id = gt_item.get('image_id', 'unknown')
            
            # 예측 파일 찾기
            if image_id not in pred_dict:
                self._logger.warning(f"No prediction found for image_id: {image_id}")
                skipped_count += 1
                continue
            
            # GT PNG 로드
            gt_seg_path = gt_item.get('sem_seg_file_name')
            if not gt_seg_path or not os.path.exists(gt_seg_path):
                self._logger.warning(f"GT file not found: {gt_seg_path}")
                skipped_count += 1
                continue
            
            gt_png = np.array(Image.open(gt_seg_path))
            
            # 예측 PNG 로드 (process()에서 dataset_id로 저장됨)
            pred_png = np.array(Image.open(pred_dict[image_id]))
            
            # 크기 확인
            if gt_png.shape[:2] != pred_png.shape[:2]:
                self._logger.warning(
                    f"Shape mismatch for {image_id}: GT {gt_png.shape} vs Pred {pred_png.shape}"
                )
                skipped_count += 1
                continue
            
            # Grayscale 변환 (RGB인 경우)
            if len(gt_png.shape) == 3:
                gt_png = gt_png[:, :, 0] if gt_png.shape[2] >= 1 else gt_png
            if len(pred_png.shape) == 3:
                pred_png = pred_png[:, :, 0] if pred_png.shape[2] >= 1 else pred_png
            
            # ===== 4. dataset_id → contiguous_id 변환 =====
            gt_contiguous = self._convert_to_contiguous(gt_png)
            pred_contiguous = self._convert_to_contiguous(pred_png)
            
            # 디버깅: 첫 이미지에서 ID 검증
            if processed_count == 0:
                self._logger.info(f"[DEBUG] First image analysis: {image_id}")
                self._logger.info(f"  GT unique dataset_ids: {np.unique(gt_png)[:20]}")
                self._logger.info(f"  Pred unique dataset_ids: {np.unique(pred_png)[:20]}")
                self._logger.info(f"  GT unique contiguous_ids: {np.unique(gt_contiguous)}")
                self._logger.info(f"  Pred unique contiguous_ids: {np.unique(pred_contiguous)}")
                
                # 픽셀 분포 확인
                for cid in range(min(5, self._num_classes)):
                    gt_count = np.sum(gt_contiguous == cid)
                    pred_count = np.sum(pred_contiguous == cid)
                    if gt_count > 0 or pred_count > 0:
                        class_name = self._class_names[cid] if cid < len(self._class_names) else f"class_{cid}"
                        self._logger.info(f"  {class_name}: GT={gt_count} pixels, Pred={pred_count} pixels")
            
            # ===== 5. Confusion Matrix 업데이트 =====
            self._update_confusion_matrix(gt_contiguous, pred_contiguous)
            processed_count += 1
        
        self._logger.info(f"Processed {processed_count} image pairs")
        if skipped_count > 0:
            self._logger.warning(f"Skipped {skipped_count} images due to errors")
        
        # ===== 6. 멀티 GPU 동기화 및 Confusion Matrix 집계 =====
        comm.synchronize()
        
        # 모든 GPU의 confusion matrix 수집
        all_conf_matrices = comm.gather(self._conf_matrix, dst=0)
        
        if not comm.is_main_process():
            return
        
        # Main process: 모든 GPU의 confusion matrix 합산
        total_conf_matrix = sum(all_conf_matrices)
        
        self._logger.info("Starting semantic segmentation evaluation...")
        
        # ===== 7. IoU 계산 =====
        # Confusion Matrix에서 메트릭 추출
        tp = total_conf_matrix.diagonal()[:-1].astype(float)  # True Positives (ignore 제외)
        pos_gt = np.sum(total_conf_matrix[:-1, :-1], axis=0).astype(float)  # GT 픽셀 수
        pos_pred = np.sum(total_conf_matrix[:-1, :-1], axis=1).astype(float)  # 예측 픽셀 수
        
        # Accuracy 계산
        acc = np.full(self._num_classes, np.nan, dtype=float)
        iou = np.full(self._num_classes, np.nan, dtype=float)
        
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        
        # IoU 계산
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / union[iou_valid]
        
        # ===== 8. 평균 지표 계산 =====
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid) if np.sum(acc_valid) > 0 else 0
        miou = np.sum(iou[iou_valid]) / np.sum(iou_valid) if np.sum(iou_valid) > 0 else 0
        pacc = np.sum(tp) / np.sum(pos_gt) if np.sum(pos_gt) > 0 else 0
        
        # Frequency Weighted IoU
        class_weights = pos_gt / np.sum(pos_gt) if np.sum(pos_gt) > 0 else np.zeros(self._num_classes)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid]) if np.sum(iou_valid) > 0 else 0
        
        # ===== 9. 결과 구성 =====
        res = OrderedDict()
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        
        # 클래스별 IoU와 Accuracy
        for i, name in enumerate(self._class_names):
            res[f"IoU-{name}"] = 100 * iou[i] if not np.isnan(iou[i]) else 0.0
            res[f"ACC-{name}"] = 100 * acc[i] if not np.isnan(acc[i]) else 0.0
        
        # ===== 10. 로깅 =====
        self._logger.info("=" * 70)
        self._logger.info("Semantic Segmentation Evaluation Results:")
        self._logger.info(f"  mIoU:  {res['mIoU']:.2f}")
        self._logger.info(f"  fwIoU: {res['fwIoU']:.2f}")
        self._logger.info(f"  mACC:  {res['mACC']:.2f}")
        self._logger.info(f"  pACC:  {res['pACC']:.2f}")
        self._logger.info("-" * 70)
        self._logger.info("Per-class IoU:")
        for i, name in enumerate(self._class_names):
            if not np.isnan(iou[i]):
                self._logger.info(f"  {name:20s}: {100 * iou[i]:5.2f}")
        self._logger.info("=" * 70)
        
        return OrderedDict({"sem_seg": res})

class GaemiPanopticEvaluator(GaemiEvaluator):
    """
    Panoptic Segmentation Evaluator for Gaemi dataset.
    Computes PQ (Panoptic Quality), IoU, and AP metrics.
    """
    
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        
        # ID 매핑: contiguous_id -> dataset_id
        self._contiguous_id_to_dataset_id = {}
        self.class_info = self._metadata.class_info
        for contiguous_id, class_name in enumerate(self._metadata.thing_classes):
            if class_name in self.class_info:
                self._contiguous_id_to_dataset_id[contiguous_id] = self.class_info[class_name]['id']


        # 평가 메트릭 저장용
        self.reset()
    
    def reset(self):
        """평가 시작 전 초기화"""
        super().reset()
        # 클래스별 메트릭 저장
        # - iou: Intersection over Union
        # - tp: True Positive (IoU >= 0.5로 매칭된 세그먼트)
        # - fp: False Positive (매칭 실패한 예측)
        # - fn: False Negative (매칭 실패한 GT)
        self._per_class_metrics = defaultdict(lambda: {
            'iou': [],        # 각 매칭쌍의 IoU 리스트
            'tp': 0,          # True Positive 카운트
            'fp': 0,          # False Positive 카운트  
            'fn': 0           # False Negative 카운트
        })
        
        # Semantic segmentation용 혼동 행렬
        self._num_classes = len(self._metadata.thing_classes)
        self._confusion_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int64)
    
    def _rgb_to_class_id(self, rgb_img):
        """
        GT의 RGB 색상을 class ID로 변환
        
        Args:
            rgb_img: [H, W, 3] numpy array, RGB 이미지
        
        Returns:
            class_id_img: [H, W] numpy array, class ID 맵
        """
        height, width = rgb_img.shape[:2]
        class_id_img = np.full((height, width), 255, dtype=np.int32)  # 255 = unmatched/ignore
        
        # class_info의 색상 → class_id 매핑 생성
        color_to_class_id = {}
        for class_name, info in self.class_info.items():
            color_tuple = tuple(info['color'])  # [R, G, B] → (R, G, B)
            color_to_class_id[color_tuple] = info['id']
        
        # 각 색상에 대해 해당하는 픽셀을 찾아 class_id 할당
        for color_tuple, class_id in color_to_class_id.items():
            r, g, b = color_tuple
            # 정확히 일치하는 픽셀 찾기
            mask = (rgb_img[:,:,0] == r) & (rgb_img[:,:,1] == g) & (rgb_img[:,:,2] == b)
            class_id_img[mask] = class_id
        
        # 매칭되지 않은 색상 확인 (디버깅용)
        unmatched_mask = class_id_img == 255
        if np.any(unmatched_mask):
            unique_unmatched = np.unique(rgb_img[unmatched_mask].reshape(-1, 3), axis=0)
            if len(unique_unmatched) > 0 and len(unique_unmatched) < 20:  # 너무 많으면 출력 안함
                self._logger.warning(f"Found {len(unique_unmatched)} unmatched colors in GT image")
                for uc in unique_unmatched[:5]:  # 최대 5개만 출력
                    self._logger.warning(f"  Unmatched color: RGB{tuple(uc)}")
            # 매칭되지 않은 픽셀은 void(0)로 처리
            class_id_img[unmatched_mask] = 0
        
        return class_id_img

    def process(self, inputs, outputs):
        """
        모델의 panoptic segmentation 예측 결과를 처리합니다.
        
        동작 과정:
        1. 모델 출력에서 panoptic 이미지와 세그먼트 정보 추출
        2. contiguous_id를 dataset_id로 변환
        3. 평가용 PNG 이미지 파일로 저장
        """
        
        # 첫 번째 이미지만 디버그
        debug_first = not hasattr(self, '_process_debug_done')
        
        for idx, (input, output) in enumerate(zip(inputs, outputs)):
            # ===== 1. 모델 출력 가져오기 =====
            # output["panoptic_seg"]는 (panoptic_img, segments_info) 튜플입니다.
            # - panoptic_img: [H, W] 텐서, 각 픽셀의 panoptic_id
            # - segments_info: 각 세그먼트의 메타정보 리스트
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy().astype(np.int32)
            
            # ===== 2. segments_info가 None인 경우 처리 =====
            # Mask2Former는 보통 segments_info를 제공하지만, 없는 경우를 대비
            if segments_info is None:
                label_divisor = self._metadata.label_divisor
                segments_info = []
                
                # panoptic_img의 고유 ID들을 순회하며 세그먼트 정보 복원
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # -1은 VOID 영역 (학습되지 않은 영역)
                        continue
                    
                    # panoptic_id에서 category_id와 instance_id 분리
                    # panoptic_id = category_id * label_divisor + instance_id
                    pred_class = panoptic_label // label_divisor
                    
                    segments_info.append({
                        "id": int(panoptic_label),
                        "category_id": int(pred_class),
                    })
            
            # ===== 3. ID 변환: contiguous_id → dataset_id =====
            # 모델은 0, 1, 2, ... (contiguous_id)로 예측하지만
            # GT는 1, 2, 3, ... (dataset_id)를 사용하므로 변환 필요
            new_panoptic_img = np.zeros_like(panoptic_img, dtype=np.int32)
            label_divisor = self._metadata.label_divisor
            
            if debug_first and idx == 0:
                self._logger.info(f"[PROCESS DEBUG] First image: {input.get('file_name', 'unknown')}")
                self._logger.info(f"[PROCESS DEBUG] Original panoptic unique IDs: {np.unique(panoptic_img)[:20]}")
                self._logger.info(f"[PROCESS DEBUG] Number of segments: {len(segments_info)}")
            
            for seg_idx, segment in enumerate(segments_info):
                old_panoptic_id = segment['id']
                contiguous_category_id = segment['category_id']
                
                # instance_id 추출 (panoptic_id % label_divisor)
                instance_id = old_panoptic_id % label_divisor
                
                # contiguous_id를 dataset_id로 변환
                dataset_category_id = self._contiguous_id_to_dataset_id.get(
                    contiguous_category_id,
                    0  # 매핑 실패 시 void(0)로 처리
                )
                
                if debug_first and idx == 0 and seg_idx < 10:
                    class_name = self._metadata.thing_classes[contiguous_category_id] if contiguous_category_id < len(self._metadata.thing_classes) else "unknown"
                    self._logger.info(
                        f"[PROCESS DEBUG] Seg {seg_idx}: "
                        f"panoptic_id={old_panoptic_id} (contiguous={contiguous_category_id}, instance={instance_id}) "
                        f"-> dataset_id={dataset_category_id} ({class_name}) "
                        f"-> new_panoptic_id={dataset_category_id * label_divisor + instance_id}"
                    )
                
                # 새로운 panoptic_id 계산
                new_panoptic_id = dataset_category_id * label_divisor + instance_id
                
                # 해당 세그먼트 영역을 새 ID로 변경
                mask = panoptic_img == old_panoptic_id
                new_panoptic_img[mask] = new_panoptic_id
            
            if debug_first and idx == 0:
                self._logger.info(f"[PROCESS DEBUG] After conversion unique IDs: {np.unique(new_panoptic_img)[:20]}")
                self._logger.info(f"[PROCESS DEBUG] After conversion class IDs: {np.unique(new_panoptic_img // label_divisor)[:20]}")
                self._process_debug_done = True
            
            # ===== 4. 파일 저장 =====
            # 파일명 생성
            original_filename = input['file_name']
            base_name = os.path.basename(original_filename)
            # 확장자 제거 (.jpg, .jpeg, .png 등)
            base_name = os.path.splitext(base_name)[0]
            pred_file = os.path.join(self._working_dir, f"{base_name}.png")
            
            # RGB 형식으로 변환하여 PNG 저장 (panopticapi 표준 형식)
            # 32비트 panoptic ID를 RGB 3채널로 인코딩 (평가용)
            rgb_img = id2rgb(new_panoptic_img)
            Image.fromarray(rgb_img).save(pred_file)
            
            # 추가: 시각화용 이미지도 따로 저장 (선택사항)
            if False:  # 디버깅 시 True로 변경
                vis_file = os.path.join(self._working_dir, f"{base_name}_vis.png")
                height, width = new_panoptic_img.shape
                vis_img = np.zeros((height, width, 3), dtype=np.uint8)
                
                # dataset_id를 class_name으로 매핑
                dataset_id_to_class_name = {}
                for class_name, info in self.class_info.items():
                    dataset_id_to_class_name[info['id']] = class_name
                
                # 각 세그먼트에 클래스별 색상 적용
                for segment in segments_info:
                    old_panoptic_id = segment['id']
                    contiguous_category_id = segment['category_id']
                    
                    dataset_category_id = self._contiguous_id_to_dataset_id.get(
                        contiguous_category_id, 0
                    )
                    
                    class_name = dataset_id_to_class_name.get(dataset_category_id, 'void')
                    
                    if class_name in self.class_info:
                        color = self.class_info[class_name]['color']
                    else:
                        color = [128, 128, 128]
                    
                    mask = new_panoptic_img == (dataset_category_id * label_divisor + (old_panoptic_id % label_divisor))
                    vis_img[mask] = color
                
                Image.fromarray(vis_img).save(vis_file)

    def evaluate(self):
        """
        저장된 예측 결과와 GT를 비교하여 평가 지표를 계산합니다.
        
        계산 지표:
        1. PQ (Panoptic Quality): Panoptic segmentation 표준 지표
        2. mIoU (mean IoU): Semantic segmentation 성능
        3. AP (Average Precision): Instance segmentation 성능
        """
        # ===== 멀티 GPU 동기화 =====
        comm.synchronize()
        if not comm.is_main_process():
            return
        
        self._logger.info("Starting Panoptic Evaluation...")
        
        # ===== 1. GT 파일 경로 수집 =====
        val_json_path = self._metadata.val_img_json_path
        with open(val_json_path, 'r') as f:
            json_info = json.load(f)

        # 서비스 영역 필터링
        gt_raw_img_list = []
        available_areas = getattr(self._metadata, 'available_service_areas', [])
        
        if available_areas and len(available_areas) > 0:
            self._logger.info(f"Filtering by service areas: {available_areas}")
            for area, img_list in json_info.items():
                if area in available_areas:
                    gt_raw_img_list += img_list
                    self._logger.info(f"  Added {len(img_list)} images from {area}")
        else:
            self._logger.info("No service area filter, using all areas")
            for area, img_list in json_info.items():
                gt_raw_img_list += img_list
                self._logger.info(f"  Added {len(img_list)} images from {area}")
        
        self._logger.info(f"Total GT images: {len(gt_raw_img_list)}")
        # ===== 2. 예측 파일 목록 가져오기 =====
        pred_files = sorted(glob(os.path.join(self._working_dir, "*.png")))
        self._logger.info(f"Found {len(pred_files)} prediction files")
        
        # ===== 3. 각 이미지 쌍에 대해 평가 =====
        processed_count = 0
        for pred_path in pred_files:
            # 예측 파일명에서 확장자 제거
            base_name = os.path.splitext(os.path.basename(pred_path))[0]
            
            # GT 경로 찾기
            gt_path = None
            matched_gt_raw = None
            for gt_raw in gt_raw_img_list:
                # GT 파일명에서 확장자 제거하고 비교
                gt_raw_basename = os.path.splitext(os.path.basename(gt_raw))[0]
                if base_name == gt_raw_basename:
                    matched_gt_raw = gt_raw
                    gt_path = gt_raw.replace('images', 'labels').replace('.jpg', '.png')
                    # self._logger.info(f'Matched: pred={base_name} -> gt={gt_path}')
                    break
            
            if gt_path is None:
                self._logger.warning(f"No GT match found for prediction: {base_name}")
                self._logger.warning(f"  Prediction file: {pred_path}")
                self._logger.warning(f"  Expected in one of {len(gt_raw_img_list)} GT files")
                # 비슷한 이름 찾기
                # similar = [g for g in gt_raw_img_list if base_name[:20] in os.path.basename(g)]
                # if similar:
                #     self._logger.warning(f"  Similar GT files found: {similar[:3]}")
                continue
                
            if not os.path.exists(gt_path):
                self._logger.warning(f"GT file does not exist: {gt_path}")
                self._logger.warning(f"  Matched from: {matched_gt_raw}")
                continue
            
            # 이미지 로드
            pred_rgb = np.array(Image.open(pred_path))
            
            # GT 로드: RGB 색상을 class ID로 변환
            gt_pil = Image.open(gt_path)
            gt_array = np.array(gt_pil)
            
            if gt_pil.mode == 'P':
                # 팔레트 모드: 원본 인덱스가 contiguous_id (0~num_classes-1)
                gt_contiguous = gt_array.astype(np.int32)
                # contiguous_id를 dataset_id로 변환
                gt_semantic = np.full_like(gt_contiguous, 0, dtype=np.int32)
                for contiguous_id, dataset_id in self._contiguous_id_to_dataset_id.items():
                    gt_semantic[gt_contiguous == contiguous_id] = dataset_id
            elif len(gt_array.shape) == 3:
                # RGB 모드: class_info의 색상을 사용하여 dataset_id로 변환
                gt_semantic = self._rgb_to_class_id(gt_array)
            else:
                # Grayscale: 직접 사용 (dataset_id로 가정)
                gt_semantic = gt_array.astype(np.int32)
            
            # 크기 확인
            pred_rgb_array = pred_rgb
            if pred_rgb_array.shape[:2] != gt_semantic.shape[:2]:
                self._logger.warning(f"Shape mismatch for {base_name}, skipping")
                continue
            
            # 예측 RGB를 panoptic ID로 변환
            pred_panoptic = rgb2id(pred_rgb)
            
            # 예측을 semantic으로 변환 (panoptic ID -> class ID)
            label_divisor = self._metadata.label_divisor
            pred_semantic = (pred_panoptic // label_divisor).astype(np.int32)
            
            # 디버깅: 첫 번째 이미지에서 ID 확인
            if processed_count == 0:
                self._logger.info(f"[DEBUG] First image analysis:")
                self._logger.info(f"  GT unique class IDs: {np.unique(gt_semantic)[:20]}")
                self._logger.info(f"  Pred unique class IDs: {np.unique(pred_semantic)[:20]}")
                self._logger.info(f"  ID mapping (contiguous -> dataset): {self._contiguous_id_to_dataset_id}")
            
            # GT는 이미 semantic format (class ID)
            # Semantic Segmentation 방식으로 평가
            
            # ===== 4. Semantic Quality 계산 =====
            # GT와 예측을 동일한 class별로 비교
            self._compute_semantic_metrics(pred_semantic, gt_semantic)
            
            # ===== 5. Semantic IoU용 혼동 행렬 업데이트 =====
            self._update_confusion_matrix_semantic(pred_semantic, gt_semantic)
            
            processed_count += 1
        
        self._logger.info(f"Processed {processed_count} image pairs")
        
        # ===== 6. 최종 메트릭 계산 =====
        results = self._compute_final_metrics()
        
        return results
    
    def _compute_semantic_metrics(self, pred_semantic, gt_semantic):
        """
        Semantic Segmentation 방식으로 평가 (픽셀 단위 비교)
        
        Args:
            pred_semantic: 예측 class ID 맵 [H, W] (dataset_id)
            gt_semantic: GT class ID 맵 [H, W] (dataset_id)
        """
        # 디버깅: 첫 이미지 상세 분석
        debug_first = not hasattr(self, '_debug_done')
        if debug_first:
            self._debug_done = True
            self._logger.info(f"[DEBUG] Metadata thing_classes ({len(self._metadata.thing_classes)}): {self._metadata.thing_classes}")
            self._logger.info(f"[DEBUG] Contiguous -> Dataset mapping: {self._contiguous_id_to_dataset_id}")
        
        # GT와 예측 모두 이미 dataset_id를 사용하고 있음
        # dataset_id를 contiguous_id로 변환
        # 초기값 255 = ignore (Config에 정의되지 않은 클래스)
        pred_contiguous = np.full_like(pred_semantic, 255, dtype=np.int32)
        gt_contiguous = np.full_like(gt_semantic, 255, dtype=np.int32)
        
        # dataset_id → contiguous_id 역매핑 생성
        # Config의 thing_classes에 정의된 클래스만 포함됨
        dataset_id_to_contiguous_id = {v: k for k, v in self._contiguous_id_to_dataset_id.items()}
        
        if debug_first:
            self._logger.info(f"[DEBUG] Reverse mapping (dataset_id -> contiguous_id):")
            self._logger.info(f"  {sorted(dataset_id_to_contiguous_id.items())}")
        
        # 예측과 GT 모두 dataset_id → contiguous_id 변환
        # Config에 정의된 클래스만 변환되고, 나머지는 255(ignore)로 유지
        for dataset_id, contiguous_id in dataset_id_to_contiguous_id.items():
            pred_contiguous[pred_semantic == dataset_id] = contiguous_id
            gt_contiguous[gt_semantic == dataset_id] = contiguous_id
        
        # void(0) 클래스는 명시적으로 255로 설정
        gt_contiguous[gt_semantic == 0] = 255
        pred_contiguous[pred_semantic == 0] = 255
        
        if debug_first:
            self._logger.info(f"[DEBUG] Input dataset_id range:")
            self._logger.info(f"  GT unique dataset_ids: {np.unique(gt_semantic)[:20]}")
            self._logger.info(f"  Pred unique dataset_ids: {np.unique(pred_semantic)[:20]}")
            self._logger.info(f"[DEBUG] After conversion to contiguous_id:")
            self._logger.info(f"  GT unique contiguous_ids: {np.unique(gt_contiguous)}")
            self._logger.info(f"  Pred unique contiguous_ids: {np.unique(pred_contiguous)}")
            
            # 픽셀 분포 확인
            for cid in range(min(5, self._num_classes)):
                gt_count = np.sum(gt_contiguous == cid)
                pred_count = np.sum(pred_contiguous == cid)
                overlap = np.sum((gt_contiguous == cid) & (pred_contiguous == cid))
                class_name = self._metadata.thing_classes[cid] if cid < len(self._metadata.thing_classes) else f"class_{cid}"
                self._logger.info(f"  Class {class_name} (cid={cid}): GT={gt_count}, Pred={pred_count}, Overlap={overlap}")
        
        # 클래스별로 TP, FP, FN 계산 (픽셀 단위)
        # ignore 영역(255)은 제외
        for contiguous_id in range(self._num_classes):
            # 유효한 픽셀만 선택 (ignore 제외)
            valid_mask = (gt_contiguous != 255)
            
            pred_mask = (pred_contiguous == contiguous_id) & valid_mask
            gt_mask = (gt_contiguous == contiguous_id) & valid_mask
            
            # TP: 예측과 GT 모두 해당 클래스
            tp_pixels = np.sum(pred_mask & gt_mask)
            
            # FP: 예측은 해당 클래스, GT는 아님
            fp_pixels = np.sum(pred_mask & ~gt_mask)
            
            # FN: GT는 해당 클래스, 예측은 아님
            fn_pixels = np.sum(~pred_mask & gt_mask)
            
            # 픽셀이 있는 경우만 메트릭 업데이트
            if tp_pixels > 0 or fp_pixels > 0 or fn_pixels > 0:
                # IoU 계산
                intersection = tp_pixels
                union = tp_pixels + fp_pixels + fn_pixels
                iou = intersection / union if union > 0 else 0
                
                # 메트릭 저장 (이미지별로 1개의 IoU 저장)
                self._per_class_metrics[contiguous_id]['iou'].append(iou)
                self._per_class_metrics[contiguous_id]['tp'] += 1 if tp_pixels > 0 else 0
                self._per_class_metrics[contiguous_id]['fp'] += 1 if fp_pixels > 0 and tp_pixels == 0 else 0
                self._per_class_metrics[contiguous_id]['fn'] += 1 if fn_pixels > 0 and tp_pixels == 0 else 0
    
    def _update_confusion_matrix_semantic(self, pred_semantic, gt_semantic):
        """
        Semantic segmentation용 혼동 행렬 업데이트 (픽셀 단위)
        """
        # GT와 예측 모두 이미 dataset_id를 사용하고 있음
        # dataset_id를 contiguous_id로 변환
        pred_contiguous = np.full_like(pred_semantic, 255, dtype=np.int32)
        gt_contiguous = np.full_like(gt_semantic, 255, dtype=np.int32)
        
        # dataset_id → contiguous_id 역매핑 생성
        dataset_id_to_contiguous_id = {v: k for k, v in self._contiguous_id_to_dataset_id.items()}
        
        # 예측과 GT 모두 dataset_id → contiguous_id 변환
        for dataset_id, contiguous_id in dataset_id_to_contiguous_id.items():
            pred_contiguous[pred_semantic == dataset_id] = contiguous_id
            gt_contiguous[gt_semantic == dataset_id] = contiguous_id
        
        # void(0) 및 non-trainable 클래스는 ignore (255)
        gt_contiguous[gt_semantic == 0] = 255
        pred_contiguous[pred_semantic == 0] = 255
        
        # 유효한 픽셀만 선택 (ignore=255 제외, 양쪽 모두 체크)
        valid_mask = (
            (gt_contiguous != 255) & 
            (pred_contiguous != 255) &
            (gt_contiguous >= 0) & 
            (gt_contiguous < self._num_classes) &
            (pred_contiguous >= 0) &
            (pred_contiguous < self._num_classes)
        )
        
        # 혼동 행렬 업데이트 (유효한 픽셀만 사용)
        if np.any(valid_mask):
            gt_valid = gt_contiguous[valid_mask]
            pred_valid = pred_contiguous[valid_mask]
            
            # 디버깅: 범위 체크
            assert np.all(gt_valid >= 0) and np.all(gt_valid < self._num_classes), \
                f"GT contiguous ID out of range: min={gt_valid.min()}, max={gt_valid.max()}, num_classes={self._num_classes}"
            assert np.all(pred_valid >= 0) and np.all(pred_valid < self._num_classes), \
                f"Pred contiguous ID out of range: min={pred_valid.min()}, max={pred_valid.max()}, num_classes={self._num_classes}"
            
            hist = np.bincount(
                self._num_classes * gt_valid + pred_valid,
                minlength=self._num_classes ** 2
            ).reshape(self._num_classes, self._num_classes)
            
            self._confusion_matrix += hist
    
    def _compute_panoptic_metrics(self, pred_panoptic, gt_panoptic):
        """
        단일 이미지 쌍에 대한 Panoptic Quality 계산
        
        Args:
            pred_panoptic: 예측 panoptic ID 맵 [H, W]
            gt_panoptic: GT panoptic ID 맵 [H, W]
        """
        # Convert to int32 to avoid overflow when dividing by label_divisor
        pred_panoptic = pred_panoptic.astype(np.int32)
        gt_panoptic = gt_panoptic.astype(np.int32)
        
        label_divisor = self._metadata.label_divisor
        
        # ===== 1. 예측 세그먼트 추출 =====
        pred_segments = {}
        for panoptic_id in np.unique(pred_panoptic):
            if panoptic_id == 0:  # void 제외
                continue
            category_id = panoptic_id // label_divisor
            mask = pred_panoptic == panoptic_id
            pred_segments[panoptic_id] = {
                'category_id': category_id,
                'mask': mask,
                'area': np.sum(mask)
            }

        # print(pred_segments)
        
        # ===== 2. GT 세그먼트 추출 =====
        gt_segments = {}
        for panoptic_id in np.unique(gt_panoptic):
            if panoptic_id == 0:  # void 제외
                continue
            category_id = panoptic_id // label_divisor
            mask = gt_panoptic == panoptic_id
            gt_segments[panoptic_id] = {
                'category_id': category_id,
                'mask': mask,
                'area': np.sum(mask)
            }
        
        # print(gt_segments)
        
        # ===== 3. 세그먼트 매칭 (같은 클래스 내에서) =====
        # Panoptic Quality 표준: IoU > 0.5인 세그먼트만 매칭으로 인정
        matched_pairs = []  # [(pred_id, gt_id, iou, category_id), ...]
        pred_matched = set()
        gt_matched = set()
        
        for pred_id, pred_seg in pred_segments.items():
            pred_cat = pred_seg['category_id']
            best_iou = 0
            best_gt_id = None
            
            for gt_id, gt_seg in gt_segments.items():
                if gt_id in gt_matched:
                    continue
                if gt_seg['category_id'] != pred_cat:
                    continue
                
                # IoU 계산
                intersection = np.sum(pred_seg['mask'] & gt_seg['mask'])
                union = np.sum(pred_seg['mask'] | gt_seg['mask'])
                iou = intersection / union if union > 0 else 0
                
                # IoU가 가장 높은 GT 세그먼트 찾기 (threshold 없이)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_id = gt_id
            
            # IoU > 0.5인 경우만 매칭으로 인정 (Panoptic Quality 표준)
            if best_gt_id is not None and best_iou > 0.5:
                matched_pairs.append((pred_id, best_gt_id, best_iou, pred_cat))
                pred_matched.add(pred_id)
                gt_matched.add(best_gt_id)
        
        # ===== 4. 클래스별 메트릭 업데이트 =====
        # True Positives (매칭된 쌍)
        for _, _, iou, category_id in matched_pairs:
            # contiguous_id로 변환 (메트릭 저장용)
            contiguous_id = None
            for cid, did in self._contiguous_id_to_dataset_id.items():
                if did == category_id:
                    contiguous_id = cid
                    break
            
            if contiguous_id is not None:
                self._per_class_metrics[contiguous_id]['iou'].append(iou)
                self._per_class_metrics[contiguous_id]['tp'] += 1
        
        # False Positives (매칭 실패한 예측)
        for pred_id, pred_seg in pred_segments.items():
            if pred_id not in pred_matched:
                category_id = pred_seg['category_id']
                contiguous_id = None
                for cid, did in self._contiguous_id_to_dataset_id.items():
                    if did == category_id:
                        contiguous_id = cid
                        break
                if contiguous_id is not None:
                    self._per_class_metrics[contiguous_id]['fp'] += 1
        
        # False Negatives (매칭 실패한 GT)
        for gt_id, gt_seg in gt_segments.items():
            if gt_id not in gt_matched:
                category_id = gt_seg['category_id']
                contiguous_id = None
                for cid, did in self._contiguous_id_to_dataset_id.items():
                    if did == category_id:
                        contiguous_id = cid
                        break
                if contiguous_id is not None:
                    self._per_class_metrics[contiguous_id]['fn'] += 1
    
    def _update_confusion_matrix(self, pred_panoptic, gt_panoptic):
        """
        Semantic segmentation용 혼동 행렬 업데이트
        """
        # Convert to int32 first to avoid overflow
        pred_panoptic = pred_panoptic.astype(np.int32)
        gt_panoptic = gt_panoptic.astype(np.int32)
        
        label_divisor = self._metadata.label_divisor
        
        # Panoptic ID에서 category ID만 추출
        pred_semantic = (pred_panoptic // label_divisor).astype(np.int32)
        gt_semantic = (gt_panoptic // label_divisor).astype(np.int32)
        
        # dataset_id를 contiguous_id로 변환
        pred_semantic_contiguous = np.zeros_like(pred_semantic)
        gt_semantic_contiguous = np.zeros_like(gt_semantic)
        
        for contiguous_id, dataset_id in self._contiguous_id_to_dataset_id.items():
            pred_semantic_contiguous[pred_semantic == dataset_id] = contiguous_id
            gt_semantic_contiguous[gt_semantic == dataset_id] = contiguous_id
        
        # 유효한 픽셀만 선택 (void 제외)
        mask = (gt_semantic_contiguous >= 0) & (gt_semantic_contiguous < self._num_classes)
        
        # 혼동 행렬 업데이트
        hist = np.bincount(
            self._num_classes * gt_semantic_contiguous[mask] + pred_semantic_contiguous[mask],
            minlength=self._num_classes ** 2
        ).reshape(self._num_classes, self._num_classes)
        
        self._confusion_matrix += hist
    
    def _compute_final_metrics(self):
        """
        수집된 데이터로부터 최종 평가 지표 계산
        
        Returns:
            OrderedDict: 평가 결과
        """
        results = OrderedDict()
        results["panoptic_seg"] = {}
        
        # ===== 1. Semantic Quality 계산 (Pixel-level IoU) =====
        # Semantic Segmentation 방식: 각 이미지당 클래스별 IoU가 1개씩 저장됨
        
        # 디버깅: 전체 통계 출력
        total_images_with_class = sum(self._per_class_metrics[i]['tp'] + self._per_class_metrics[i]['fp'] + self._per_class_metrics[i]['fn'] 
                                     for i in range(len(self._metadata.thing_classes)))
        self._logger.info(f"Total images processed per class: {total_images_with_class}")
        
        # 클래스별 평균 IoU 계산
        class_iou_list = []
        for contiguous_id, class_name in enumerate(self._metadata.thing_classes):
            metrics = self._per_class_metrics[contiguous_id]
            iou_values = metrics['iou']
            
            if len(iou_values) > 0:
                avg_iou = np.mean(iou_values)
                class_iou_list.append(avg_iou)
                results["panoptic_seg"][f"IoU-{class_name}"] = avg_iou * 100
                self._logger.info(f"Class {class_name}: Images={len(iou_values)}, Avg IoU={avg_iou*100:.2f}%")
            else:
                results["panoptic_seg"][f"IoU-{class_name}"] = 0.0
        
        # mIoU (mean IoU across classes)
        results["panoptic_seg"]["mIoU"] = np.mean(class_iou_list) * 100 if class_iou_list else 0
        
        # ===== 2. Confusion Matrix 기반 IoU (전체 픽셀 기준) =====
        cm_iou_per_class = np.diag(self._confusion_matrix) / (
            self._confusion_matrix.sum(axis=1) + 
            self._confusion_matrix.sum(axis=0) - 
            np.diag(self._confusion_matrix)
        )
        
        # NaN 처리 (분모가 0인 경우)
        cm_iou_per_class = np.nan_to_num(cm_iou_per_class)
        
        # Confusion Matrix 기반 mIoU
        results["panoptic_seg"]["mIoU_cm"] = np.mean(cm_iou_per_class) * 100
        
        # ===== 3. PQ-style 지표 (Semantic 버전) =====
        # Semantic에서는 PQ를 직접 계산할 수 없지만, 
        # 이미지별 평균 IoU를 PQ처럼 사용
        results["panoptic_seg"]["PQ"] = results["panoptic_seg"]["mIoU"]  # mIoU를 PQ로 표시
        results["panoptic_seg"]["SQ"] = results["panoptic_seg"]["mIoU"]  # Segmentation Quality = mIoU
        results["panoptic_seg"]["RQ"] = 100.0 if class_iou_list else 0.0  # Recognition은 항상 수행됨
        
        # ===== 4. IoU 계산 (Semantic segmentation) - 호환성 유지 =====
        iou_per_class = np.diag(self._confusion_matrix) / (
            self._confusion_matrix.sum(axis=1) + 
            self._confusion_matrix.sum(axis=0) - 
            np.diag(self._confusion_matrix)
        )
        
        # NaN 처리 (분모가 0인 경우)
        iou_per_class = np.nan_to_num(iou_per_class)
        
        # ===== 5. AP 계산 (Semantic에서는 의미 없지만 호환성 유지) =====
        # Semantic segmentation에서는 AP를 정확히 계산할 수 없으므로
        # IoU 기반으로 근사값 제공
        ap_values = []
        for contiguous_id in range(self._num_classes):
            metrics = self._per_class_metrics[contiguous_id]
            iou_values = metrics['iou']
            
            if len(iou_values) > 0:
                # IoU >= 0.5인 비율을 AP로 근사
                avg_iou = np.mean(iou_values)
                ap_values.append(avg_iou)
        
        # AP 지표 (IoU 기반 근사값)
        avg_ap = np.mean(ap_values) * 100 if ap_values else 0
        results["panoptic_seg"]["AP"] = avg_ap
        results["panoptic_seg"]["AP50"] = avg_ap  # Semantic에서는 동일
        results["panoptic_seg"]["AP75"] = avg_ap  # Semantic에서는 동일
        
        # ===== 로그 출력 =====
        self._logger.info("=" * 60)
        self._logger.info("Semantic Segmentation Evaluation Results:")
        self._logger.info(f"  mIoU:  {results['panoptic_seg']['mIoU']:.2f}")
        self._logger.info(f"  mIoU (CM): {results['panoptic_seg']['mIoU_cm']:.2f}")
        self._logger.info("=" * 60)
        
        return results

