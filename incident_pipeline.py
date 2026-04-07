from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime
    YOLO = None

try:
    from tensorflow.keras.layers import GRU, Dense, Input
    from tensorflow.keras.models import Sequential
except ImportError:  # pragma: no cover - handled at runtime
    GRU = Dense = Input = Sequential = None


VEHICLE_CLASS_IDS = {2, 3, 5, 7}
COCO_LABELS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class VehicleDetection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class IncidentState:
    vehicles: List[VehicleDetection]
    fire_regions: List[Tuple[int, int, int, int]]
    temporal_probability: float
    triggered: bool
    summary: str


class TemporalCrashModel:
    """Uses a lightweight GRU when weights are available, otherwise a heuristic sequence scorer."""

    def __init__(self, sequence_length: int = 16, weights_path: Optional[str] = None) -> None:
        self.sequence_length = sequence_length
        self.model = None

        if Sequential is not None and weights_path:
            self.model = self._build_model()
            self.model.load_weights(weights_path)

    def _build_model(self):
        model = Sequential(
            [
                Input(shape=(self.sequence_length, 3)),
                GRU(32, return_sequences=False),
                Dense(16, activation="relu"),
                Dense(1, activation="sigmoid"),
            ]
        )
        return model

    def score(self, sequence: Sequence[Sequence[float]]) -> float:
        if len(sequence) < self.sequence_length:
            return 0.0

        seq_array = np.array(sequence[-self.sequence_length :], dtype=np.float32)
        if self.model is not None:
            prediction = self.model.predict(seq_array[np.newaxis, ...], verbose=0)[0][0]
            return float(prediction)

        # Heuristic fallback: sequence-aware scoring until trained GRU weights are added.
        feature_means = seq_array.mean(axis=0)
        motion, vehicle_density, fire_density = feature_means
        score = (
            0.35 * motion
            + 0.30 * vehicle_density
            + 0.35 * fire_density
        )
        return float(np.clip(score, 0.0, 1.0))


class IncidentUnderstandingPipeline:
    def __init__(
        self,
        sequence_length: int = 16,
        temporal_weights_path: Optional[str] = None,
        temporal_threshold: float = 0.6,
        detection_model_name: str = "yolov8n.pt",
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics is required for YOLOv8 detection. Install dependencies from requirements.txt."
            )

        self.detector = YOLO(detection_model_name)
        self.temporal_model = TemporalCrashModel(
            sequence_length=sequence_length,
            weights_path=temporal_weights_path,
        )
        self.temporal_threshold = temporal_threshold
        self.sequence_length = sequence_length
        self.feature_history: Deque[List[float]] = deque(maxlen=sequence_length)
        self.previous_gray: Optional[np.ndarray] = None

    def analyze(self, frame: np.ndarray) -> IncidentState:
        detections = self._detect_vehicles(frame)
        fire_regions = self._detect_fire(frame)
        motion_score = self._motion_score(frame)

        frame_features = self._build_features(
            motion_score=motion_score,
            vehicles=detections,
            fire_regions=fire_regions,
        )
        self.feature_history.append(frame_features)
        temporal_probability = self.temporal_model.score(list(self.feature_history))

        triggered = temporal_probability >= self.temporal_threshold and bool(fire_regions)
        summary = self._build_summary(
            detections,
            fire_regions,
            temporal_probability,
        )

        return IncidentState(
            vehicles=detections,
            fire_regions=fire_regions,
            temporal_probability=temporal_probability,
            triggered=triggered,
            summary=summary,
        )

    def _detect_vehicles(self, frame: np.ndarray) -> List[VehicleDetection]:
        results = self.detector.predict(frame, verbose=False, conf=0.3)
        if not results:
            return []

        vehicles: List[VehicleDetection] = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASS_IDS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            vehicles.append(
                VehicleDetection(
                    label=COCO_LABELS.get(class_id, str(class_id)),
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                )
            )

        return vehicles

    def _detect_fire(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        fire_mask = cv2.inRange(hsv, np.array([0, 120, 140]), np.array([35, 255, 255]))
        return self._mask_to_regions(fire_mask, min_area=800)

    def _motion_score(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return 0.0

        diff = cv2.absdiff(self.previous_gray, gray)
        self.previous_gray = gray
        motion_score = float(np.mean(diff) / 255.0)
        return motion_score

    def _build_features(
        self,
        motion_score: float,
        vehicles: Sequence[VehicleDetection],
        fire_regions: Sequence[Tuple[int, int, int, int]],
    ) -> List[float]:
        return [
            float(np.clip(motion_score, 0.0, 1.0)),
            float(min(len(vehicles) / 10.0, 1.0)),
            float(min(len(fire_regions) / 3.0, 1.0)),
        ]

    def _build_summary(
        self,
        vehicles: Sequence[VehicleDetection],
        fire_regions: Sequence[Tuple[int, int, int, int]],
        temporal_probability: float,
    ) -> str:
        signals = []
        if vehicles:
            labels = ", ".join(sorted({vehicle.label for vehicle in vehicles}))
            signals.append(f"vehicles: {labels}")
        if fire_regions:
            signals.append(f"fire regions: {len(fire_regions)}")

        if not signals:
            signals.append("no critical visual signals")

        return f"Temporal accident probability {temporal_probability:.2f}; " + "; ".join(signals)

    @staticmethod
    def _mask_to_regions(mask: np.ndarray, min_area: int) -> List[Tuple[int, int, int, int]]:
        regions: List[Tuple[int, int, int, int]] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            regions.append((x, y, x + width, y + height))
        return regions

def draw_incident_overlays(frame: np.ndarray, state: IncidentState) -> np.ndarray:
    output = frame.copy()

    for vehicle in state.vehicles:
        x1, y1, x2, y2 = vehicle.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (80, 170, 255), 2)
        cv2.putText(
            output,
            f"{vehicle.label} {vehicle.confidence:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 170, 255),
            2,
        )

    for region in state.fire_regions:
        x1, y1, x2, y2 = region
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(
            output,
            "fire",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 140, 255),
            2,
        )

    status_text = "ALERT" if state.triggered else "Monitoring"
    status_color = (0, 0, 255) if state.triggered else (0, 200, 0)
    cv2.rectangle(output, (0, 0), (640, 80), (20, 20, 20), -1)
    cv2.putText(
        output,
        f"{status_text} | Temporal score: {state.temporal_probability:.2f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )
    cv2.putText(
        output,
        state.summary[:95],
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return output
