from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO

from src.dataset.labeling.enrich_labels import classify_tl_color
from src.interfaces.perception_types import DetectedObject, ObjectClass

_YOLO_TO_OBJECTCLASS: dict[int, ObjectClass] = {
    0: ObjectClass.VEHICLE,
    1: ObjectClass.WALKER,
    2: ObjectClass.RED_LIGHT,
    3: ObjectClass.YELLOW_LIGHT,
    4: ObjectClass.GREEN_LIGHT,
    5: ObjectClass.SPEED_30,
    6: ObjectClass.SPEED_40,
    7: ObjectClass.SPEED_60,
    8: ObjectClass.SPEED_90,
    9: ObjectClass.STOP,
    10: ObjectClass.YIELD,
}

_LIGHT_IDS = {2, 3, 4}

_TL_FINAL_TO_OC = {
    2: ObjectClass.RED_LIGHT,
    3: ObjectClass.YELLOW_LIGHT,
    4: ObjectClass.GREEN_LIGHT,
}


class YoloDetector:
    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        device: str = "cpu",
    ) -> None:
        self.model = YOLO(weights_path)
        self.model.to(device)

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        results = self.model(image, verbose=False)
        if not results or results[0].boxes is None:
            return []

        detections: list[DetectedObject] = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].cpu())
            if cls_id not in _YOLO_TO_OBJECTCLASS:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().int().tolist()
            conf = float(box.conf[0].cpu())

            if cls_id in _LIGHT_IDS:
                if x2 <= x1 or y2 <= y1:
                    continue
                bgr_crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
                final_id, _ = classify_tl_color(bgr_crop)
                if final_id is None:
                    continue
                obj_class = _TL_FINAL_TO_OC[final_id]
            else:
                obj_class = _YOLO_TO_OBJECTCLASS[cls_id]

            detections.append(
                DetectedObject(
                    class_name=obj_class,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                )
            )

        return detections
