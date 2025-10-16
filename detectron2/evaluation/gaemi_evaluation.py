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
from Mask2Former.custom_util.config.class_config import class_info

# Panoptic evaluation을 위한 유틸리티
try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    # panopticapi가 없는 경우 간단한 구현 제공
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


class GaemiEvaluator:
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
        
        # 멀티 프로세스 환경에서 안전하게 디렉토리 생성
        # 메인 프로세스만 디렉토리를 초기화
        # if comm.is_main_process():
        #     if os.path.exists(self._working_dir):
        #         shutil.rmtree(self._working_dir)
        os.makedirs(self._working_dir, exist_ok=True)
        
        # 모든 프로세스가 디렉토리 생성을 기다림
        comm.synchronize()

    def reset(self):
        pass

    def process(self, inputs, outputs):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    # def __del__(self):
    #     if os.path.exists(self._working_dir):
    #         shutil.rmtree(self._working_dir)

class GaemiSemsegEvaluator(GaemiEvaluator):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self._contiguous_id_to_dataset_id = {}
        # thing_classes의 순서(index)가 모델의 contiguous_id 입니다.
        for contiguous_id, class_name in enumerate(self._metadata.thing_classes):
            if class_name in class_info:
                self._contiguous_id_to_dataset_id[contiguous_id] = class_info[class_name]['id']
            else:
                self._logger.warning(
                    f"Class '{class_name}' from dataset metadata not found in class_info. "
                    f"Predictions for this class may not be evaluated correctly."
                )

    def process(self, inputs, outputs):
        # 작업 디렉토리가 존재하는지 확인 (멀티 프로세스 안전)
        os.makedirs(self._working_dir, exist_ok=True)
        
        for input, output in zip(inputs, outputs):
            file_name = input['image_id']
            pred_file = os.path.join(self._working_dir, f"{file_name}.png")

            if 'sem_seg' not in output:
                raise ValueError("Expected 'sem_seg' in model output for semantic segmentation.")

            output = output['sem_seg'].argmax(dim=0).to(self._cpu_device).numpy().astype(np.uint8)
            pred = np.zeros(output.shape, dtype=np.uint8)

            # 미리 만들어둔 매핑을 사용하여 contiguous_id를 실제 dataset_id로 변환합니다.
            for contiguous_id, dataset_id in self._contiguous_id_to_dataset_id.items():
                pred[output == contiguous_id] = dataset_id

            Image.fromarray(pred).save(pred_file)

    def evaluate(self):
        comm.synchronize()
        if not comm.is_main_process():
            return

        self._logger.info("Starting evaluation...")

        val_img_json_path = self._metadata.val_img_json_path
        available_service_areas = self._metadata.available_service_areas

        with open(val_img_json_path, 'r') as f:
            json_info = json.load(f)

        gt_raw_img_list = []
        for area, img_list in json_info.items():
            if area in available_service_areas:
                gt_raw_img_list += img_list

        pred_img_list = glob(os.path.join(self._working_dir, "*.png"))
        gt_img_list = [img_path.replace('images', 'labels').replace('.jpg', '.png') for img_path in gt_raw_img_list]

        for pred_path in pred_img_list:
            basename = os.path.basename(pred_path).replace('.png', '')
            if basename in gt_img_list:
                # get gt img path
                exist_index = gt_img_list.index(basename)
                gt_img_path = gt_img_list[exist_index]
                # make img to array
                pred_img = np.array(Image.open(pred_path))
                gt_img = np.array(Image.open(gt_img_path))
                # check size
                if pred_img.shape != gt_img.shape:
                    self._logger.warning(
                        f"Size mismatch for {basename}: "
                        f"predicted size {pred_img.shape}, ground truth size {gt_img.shape}. "
                        f"Skipping this image."
                    )
                    continue
                # check unique values

class GaemiPanopticEvaluator(GaemiEvaluator):
    """
    Panoptic Segmentation Evaluator for Gaemi dataset.
    Computes PQ (Panoptic Quality), IoU, and AP metrics.
    """
    
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        
        # ID 매핑: contiguous_id -> dataset_id
        self._contiguous_id_to_dataset_id = {}
        for contiguous_id, class_name in enumerate(self._metadata.thing_classes):
            if class_name in class_info:
                self._contiguous_id_to_dataset_id[contiguous_id] = class_info[class_name]['id']
        
        
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

    def process(self, inputs, outputs):
        """
        모델의 panoptic segmentation 예측 결과를 처리합니다.
        
        동작 과정:
        1. 모델 출력에서 panoptic 이미지와 세그먼트 정보 추출
        2. contiguous_id를 dataset_id로 변환
        3. 평가용 PNG 이미지 파일로 저장
        """
        
        for input, output in zip(inputs, outputs):
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
            
            for segment in segments_info:
                old_panoptic_id = segment['id']
                contiguous_category_id = segment['category_id']
                
                # instance_id 추출 (panoptic_id % label_divisor)
                instance_id = old_panoptic_id % label_divisor
                
                # contiguous_id를 dataset_id로 변환
                dataset_category_id = self._contiguous_id_to_dataset_id.get(
                    contiguous_category_id,
                    0  # 매핑 실패 시 void(0)로 처리
                )
                
                # 새로운 panoptic_id 계산
                new_panoptic_id = dataset_category_id * label_divisor + instance_id
                
                # 해당 세그먼트 영역을 새 ID로 변경
                mask = panoptic_img == old_panoptic_id
                new_panoptic_img[mask] = new_panoptic_id
            
            # ===== 4. 파일 저장 =====
            # 파일명 생성
            base_name = os.path.basename(input['file_name']).replace('.jpg', '')
            pred_file = os.path.join(self._working_dir, f"{base_name}.png")
            
            # RGB 형식으로 변환하여 PNG 저장 (panopticapi 표준 형식)
            # 32비트 ID를 RGB 3채널로 인코딩
            rgb_img = id2rgb(new_panoptic_img)
            Image.fromarray(rgb_img).save(pred_file)

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
        if self._metadata.available_service_areas != []:
            for area, img_list in json_info.items():
                if area in self._metadata.available_service_areas:
                    gt_raw_img_list += img_list
        else:
            for area, img_list in json_info.items():
                gt_raw_img_list += img_list
        
        # ===== 2. 예측 파일 목록 가져오기 =====
        pred_files = sorted(glob(os.path.join(self._working_dir, "*.png")))
        self._logger.info(f"Found {len(pred_files)} prediction files")
        
        # ===== 3. 각 이미지 쌍에 대해 평가 =====
        processed_count = 0
        for pred_path in pred_files:
            base_name = os.path.basename(pred_path).replace('.png', '')
            
            # GT 경로 찾기
            gt_path = None
            for gt_raw in gt_raw_img_list:
                if base_name in os.path.basename(gt_raw):
                    gt_path = gt_raw.replace('images', 'labels').replace('.jpg', '.png')
                    break
            
            if gt_path is None or not os.path.exists(gt_path):
                self._logger.warning(f"GT not found for {base_name}, skipping")
                continue
            
            # 이미지 로드
            pred_rgb = np.array(Image.open(pred_path))
            
            # GT 로드: 팔레트 PNG인 경우 원본 인덱스 값 읽기
            gt_pil = Image.open(gt_path)
            if gt_pil.mode == 'P':
                # 팔레트 모드: 원본 인덱스가 class ID
                gt_semantic = np.array(gt_pil).astype(np.int32)
            elif len(np.array(gt_pil).shape) == 3:
                # RGB 모드: rgb2id로 변환
                gt_semantic = rgb2id(np.array(gt_pil))
            else:
                # Grayscale: 직접 사용
                gt_semantic = np.array(gt_pil).astype(np.int32)
            
            # 크기 확인
            pred_rgb_array = pred_rgb
            if pred_rgb_array.shape[:2] != gt_semantic.shape[:2]:
                self._logger.warning(f"Shape mismatch for {base_name}, skipping")
                continue
            
            # GT를 panoptic format으로 변환 (class ID -> panoptic ID)
            # GT는 semantic segmentation format (class ID만 있음)
            # 각 class의 연결된 영역을 개별 인스턴스로 변환
            from scipy import ndimage
            label_divisor = self._metadata.label_divisor
            gt_panoptic = np.zeros_like(gt_semantic, dtype=np.int32)
            
            for class_id in np.unique(gt_semantic):
                if class_id == 0:  # void 제외
                    continue
                    
                # 해당 클래스의 마스크
                class_mask = (gt_semantic == class_id)
                
                # 연결된 영역 찾기 (connected components)
                labeled_mask, num_instances = ndimage.label(class_mask)
                
                # 각 인스턴스에 panoptic ID 할당
                for instance_id in range(1, num_instances + 1):
                    instance_mask = (labeled_mask == instance_id)
                    panoptic_id = class_id * label_divisor + instance_id
                    gt_panoptic[instance_mask] = panoptic_id
            
            # 예측 RGB를 panoptic ID로 변환
            pred_panoptic = rgb2id(pred_rgb)
            
            # ===== 4. Panoptic Quality 계산 =====
            self._compute_panoptic_metrics(pred_panoptic, gt_panoptic)
            
            # ===== 5. Semantic IoU용 혼동 행렬 업데이트 =====
            self._update_confusion_matrix(pred_panoptic, gt_panoptic)
            
            processed_count += 1
        
        self._logger.info(f"Processed {processed_count} image pairs")
        
        # ===== 6. 최종 메트릭 계산 =====
        results = self._compute_final_metrics()
        
        return results
    
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
        matched_pairs = []  # [(pred_id, gt_id, iou), ...]
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
                
                if iou > 0.5 and iou > best_iou:  # IoU > 0.5 threshold
                    best_iou = iou
                    best_gt_id = gt_id
            
            if best_gt_id is not None:
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
        
        # ===== 1. Panoptic Quality (PQ) 계산 =====
        pq_list = []
        sq_list = []
        rq_list = []
        
        for contiguous_id, class_name in enumerate(self._metadata.thing_classes):
            metrics = self._per_class_metrics[contiguous_id]
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            iou_sum = sum(metrics['iou'])
            
            # SQ (Segmentation Quality) = average IoU of matched segments
            sq = iou_sum / tp if tp > 0 else 0
            
            # RQ (Recognition Quality) = TP / (TP + 0.5*FP + 0.5*FN)
            rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0
            
            # PQ = SQ * RQ
            pq = sq * rq
            
            if tp + fp + fn > 0:  # 해당 클래스가 존재하는 경우만
                pq_list.append(pq)
                sq_list.append(sq)
                rq_list.append(rq)
                results["panoptic_seg"][f"PQ-{class_name}"] = pq * 100
        
        # 전체 평균
        results["panoptic_seg"]["PQ"] = np.mean(pq_list) * 100 if pq_list else 0
        results["panoptic_seg"]["SQ"] = np.mean(sq_list) * 100 if sq_list else 0
        results["panoptic_seg"]["RQ"] = np.mean(rq_list) * 100 if rq_list else 0
        
        # ===== 2. IoU 계산 (Semantic segmentation) =====
        iou_per_class = np.diag(self._confusion_matrix) / (
            self._confusion_matrix.sum(axis=1) + 
            self._confusion_matrix.sum(axis=0) - 
            np.diag(self._confusion_matrix)
        )
        
        # NaN 처리 (분모가 0인 경우)
        iou_per_class = np.nan_to_num(iou_per_class)
        
        # 클래스별 IoU
        for contiguous_id, class_name in enumerate(self._metadata.thing_classes):
            results["panoptic_seg"][f"IoU-{class_name}"] = iou_per_class[contiguous_id] * 100
        
        # mIoU (mean IoU)
        results["panoptic_seg"]["mIoU"] = np.mean(iou_per_class) * 100
        
        # ===== 3. AP 계산 (Instance segmentation) =====
        # AP@0.5, AP@0.75, AP@[0.5:0.95]
        ap_50_list = []
        ap_75_list = []
        ap_list = []
        
        for contiguous_id in range(self._num_classes):
            metrics = self._per_class_metrics[contiguous_id]
            iou_values = metrics['iou']
            tp = metrics['tp']
            fp = metrics['fp']
            
            if tp + fp == 0:
                continue
            
            # AP@0.5: IoU >= 0.5인 경우를 TP로 간주
            tp_50 = len([iou for iou in iou_values if iou >= 0.5])
            ap_50 = tp_50 / (tp + fp) if (tp + fp) > 0 else 0
            ap_50_list.append(ap_50)
            
            # AP@0.75: IoU >= 0.75인 경우를 TP로 간주
            tp_75 = len([iou for iou in iou_values if iou >= 0.75])
            ap_75 = tp_75 / (tp + fp) if (tp + fp) > 0 else 0
            ap_75_list.append(ap_75)
            
            # AP (average over IoU thresholds 0.5:0.95)
            ap_thresholds = []
            for threshold in np.arange(0.5, 1.0, 0.05):
                tp_t = len([iou for iou in iou_values if iou >= threshold])
                ap_t = tp_t / (tp + fp) if (tp + fp) > 0 else 0
                ap_thresholds.append(ap_t)
            ap_list.append(np.mean(ap_thresholds))
        
        results["panoptic_seg"]["AP50"] = np.mean(ap_50_list) * 100 if ap_50_list else 0
        results["panoptic_seg"]["AP75"] = np.mean(ap_75_list) * 100 if ap_75_list else 0
        results["panoptic_seg"]["AP"] = np.mean(ap_list) * 100 if ap_list else 0
        
        # ===== 로그 출력 =====
        self._logger.info("=" * 60)
        self._logger.info("Panoptic Evaluation Results:")
        self._logger.info(f"  PQ:    {results['panoptic_seg']['PQ']:.2f}")
        self._logger.info(f"  SQ:    {results['panoptic_seg']['SQ']:.2f}")
        self._logger.info(f"  RQ:    {results['panoptic_seg']['RQ']:.2f}")
        self._logger.info(f"  mIoU:  {results['panoptic_seg']['mIoU']:.2f}")
        self._logger.info(f"  AP:    {results['panoptic_seg']['AP']:.2f}")
        self._logger.info(f"  AP50:  {results['panoptic_seg']['AP50']:.2f}")
        self._logger.info(f"  AP75:  {results['panoptic_seg']['AP75']:.2f}")
        self._logger.info("=" * 60)
        
        return results
