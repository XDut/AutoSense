"""CPU-first smart vehicle detection with Gemini-powered vehicle metadata."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import cvzone
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

VIDEO_PATH = Path(os.getenv("VIDEO_PATH", PROJECT_ROOT / "notebooks" / "circulation.mp4"))
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo12n.pt")
OUTPUT_FOLDER = Path(os.getenv("OUTPUT_FOLDER", PROJECT_ROOT / "cropped_vehicles"))
GEMINI_MODEL = "gemini-3.5-flash-lite"
ALLOWED_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

# CPU-friendly defaults. Each value can be tuned in .env for a particular CPU.
CPU_THREADS = max(1, int(os.getenv("CPU_THREADS", str(min(os.cpu_count() or 1, 8)))))
INFERENCE_SIZE = max(320, int(os.getenv("INFERENCE_SIZE", "640")))
FRAME_STRIDE = max(1, int(os.getenv("FRAME_STRIDE", "2")))
MAX_DISPLAY_FPS = max(1.0, float(os.getenv("MAX_DISPLAY_FPS", "30")))
GEMINI_WORKERS = max(1, int(os.getenv("GEMINI_WORKERS", "2")))
DISPLAY_SIZE = (1020, 600)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CountryEstimate(BaseModel):
    country: str = Field(description="Most likely country visible in the scene, or Unknown")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence from 0 to 1")


class VehicleDetails(BaseModel):
    vehicle_type: str = Field(description="Car, truck, bus, motorcycle, or bicycle")
    vehicle_color: str = Field(description="Dominant exterior color")
    vehicle_company: str = Field(description="Vehicle manufacturer, or Unknown")
    vehicle_model: str = Field(description="Vehicle model, or Unknown")
    estimated_year_built: str = Field(
        description="Best estimated model year or year range, or Unknown"
    )


class SmartVehicleDetector:
    """Detect and track vehicles while keeping playback responsive on a CPU."""

    def __init__(
        self,
        video_file: str | Path = VIDEO_PATH,
        yolo_model_path: str = YOLO_MODEL_PATH,
        output_json_path: str | Path | None = None,
        api_key: str | None = None,
    ) -> None:
        gemini_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("Set GOOGLE_API_KEY in the project .env file before running.")

        torch.set_num_threads(CPU_THREADS)
        cv2.setNumThreads(max(1, CPU_THREADS // 2))

        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self._load_yolo_model(yolo_model_path)
        self.cap = self._open_video_file(Path(video_file))
        self.area = np.array([(420, 407), (382, 448), (940, 456), (930, 419)], np.int32)
        self.processed_track_ids: set[int] = set()
        self.detected_country = "Unknown"
        self.country_confidence = 0.0
        self._cached_detections: list[tuple[list[int], int, str]] = []
        self._json_lock = threading.Lock()
        self._analysis_pool = ThreadPoolExecutor(
            max_workers=GEMINI_WORKERS,
            thread_name_prefix="gemini-vehicle",
        )

        current_date = time.strftime("%Y-%m-%d")
        self.output_json_path = Path(
            output_json_path or PROJECT_ROOT / f"vehicle_data_{current_date}.json"
        )
        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.vehicle_data_list = self._load_existing_json()

        self.cropped_images_folder = OUTPUT_FOLDER
        self.cropped_images_folder.mkdir(parents=True, exist_ok=True)

    def _load_yolo_model(self, path: str) -> None:
        try:
            self.yolo_model = YOLO(path)
            self.names = self.yolo_model.names
            self.device = "cpu"
            self.yolo_model.to(self.device)
            logging.info(
                "YOLO model %s loaded on CPU (%s threads, inference size %s)",
                path,
                CPU_THREADS,
                INFERENCE_SIZE,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model: {exc}") from exc

    @staticmethod
    def _open_video_file(path: Path) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {path}")
        return cap

    def _load_existing_json(self) -> list[dict]:
        if not self.output_json_path.exists():
            self._write_json([])
            logging.info("JSON output initialized: %s", self.output_json_path)
            return []

        try:
            data = json.loads(self.output_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Existing output is not valid JSON: {self.output_json_path}"
            ) from exc
        if not isinstance(data, list):
            raise ValueError(f"Existing output must contain a JSON array: {self.output_json_path}")
        return data

    def _write_json(self, data: list[dict]) -> None:
        """Atomically replace the JSON file so readers never see a partial document."""
        temporary_path = self.output_json_path.with_suffix(self.output_json_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self.output_json_path)

    @staticmethod
    def _encode_jpeg(image: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ValueError("Could not encode image as JPEG")
        return buffer.tobytes()

    def _generate_structured(self, prompt: str, image: np.ndarray, schema: type[BaseModel]):
        response = self.gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(data=self._encode_jpeg(image), mime_type="image/jpeg"),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.1,
                max_output_tokens=256,
            ),
        )
        if isinstance(response.parsed, schema):
            return response.parsed
        if response.parsed is not None:
            return schema.model_validate(response.parsed)
        if not response.text:
            raise ValueError("Gemini returned an empty response")
        return schema.model_validate_json(response.text)

    def detect_country_from_scene(self, frame: np.ndarray) -> CountryEstimate:
        """Estimate scene country once and reuse it for every vehicle entry."""
        prompt = (
            "Estimate the most likely country shown in this road scene using road signs, "
            "license-plate conventions, traffic direction, vehicles, and surroundings. "
            "Use 'Unknown' when the evidence is insufficient."
        )
        try:
            estimate = self._generate_structured(prompt, frame, CountryEstimate)
            logging.info(
                "Country detected: %s (confidence %.2f)",
                estimate.country,
                estimate.confidence,
            )
            return estimate
        except Exception as exc:
            logging.error("Country detection failed: %s", exc)
            return CountryEstimate(country="Unknown", confidence=0.0)

    def analyze_vehicle(self, image: np.ndarray) -> VehicleDetails:
        prompt = (
            f"Analyze the single cropped vehicle. The surrounding scene is most likely in "
            f"{self.detected_country}. Identify only details supported by the image. Use "
            "'Unknown' for an unreadable manufacturer/model/year. For the build year, return "
            "a single estimated year or a concise range such as '2018-2021'."
        )
        return self._generate_structured(prompt, image, VehicleDetails)

    def _relative_crop_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path)

    def process_crop_image(
        self,
        image: np.ndarray,
        track_id: int,
        detector_class: str,
        timestamp: str,
    ) -> None:
        """Analyze a crop and append exactly one schema-stable JSON record."""
        filename_timestamp = timestamp.replace(":", "-")
        image_filename = self.cropped_images_folder / f"vehicle_{track_id}_{filename_timestamp}.jpg"
        cv2.imwrite(str(image_filename), image)

        status = "complete"
        try:
            details = self.analyze_vehicle(image)
        except Exception as exc:
            logging.error("Vehicle analysis failed for track %s: %s", track_id, exc)
            details = VehicleDetails(
                vehicle_type=detector_class,
                vehicle_color="Unknown",
                vehicle_company="Unknown",
                vehicle_model="Unknown",
                estimated_year_built="Unknown",
            )
            status = "failed"

        vehicle_type = details.vehicle_type
        if not vehicle_type or vehicle_type.casefold() == "unknown":
            vehicle_type = detector_class

        entry = {
            "timestamp": timestamp,
            "track_id": int(track_id),
            "vehicle_type": vehicle_type,
            "vehicle_color": details.vehicle_color or "Unknown",
            "vehicle_company": details.vehicle_company or "Unknown",
            "vehicle_model": details.vehicle_model or "Unknown",
            "estimated_year_built": details.estimated_year_built or "Unknown",
            "country": self.detected_country,
            "country_confidence": round(self.country_confidence, 3),
            "detector_class": detector_class,
            "analysis_model": GEMINI_MODEL,
            "analysis_status": status,
            "crop_path": self._relative_crop_path(image_filename),
        }
        with self._json_lock:
            self.vehicle_data_list.append(entry)
            self._write_json(self.vehicle_data_list)
        logging.info("JSON entry added for track ID %s", track_id)

    def crop_and_process(
        self,
        clean_frame: np.ndarray,
        box: list[int],
        track_id: int,
        detector_class: str,
    ) -> None:
        if track_id < 0 or track_id in self.processed_track_ids:
            return

        height, width = clean_frame.shape[:2]
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
        y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
        if x2 <= x1 or y2 <= y1:
            return

        cropped_image = clean_frame[y1:y2, x1:x2].copy()
        timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
        self.processed_track_ids.add(track_id)
        self._analysis_pool.submit(
            self.process_crop_image,
            cropped_image,
            track_id,
            detector_class,
            timestamp,
        )

    def _run_detection(self, frame: np.ndarray) -> None:
        with torch.inference_mode():
            results = self.yolo_model.track(
                frame,
                persist=True,
                device="cpu",
                imgsz=INFERENCE_SIZE,
                half=False,
                verbose=False,
            )

        detections: list[tuple[list[int], int, str]] = []
        if results and results[0].boxes is not None:
            result_boxes = results[0].boxes
            boxes = result_boxes.xyxy.int().cpu().tolist()
            class_ids = result_boxes.cls.int().cpu().tolist()
            track_ids = (
                result_boxes.id.int().cpu().tolist()
                if result_boxes.id is not None
                else [-1] * len(boxes)
            )
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                class_name = self.names[class_id]
                if class_name in ALLOWED_CLASSES:
                    detections.append((box, int(track_id), class_name))
        self._cached_detections = detections

    def process_video_frame(self, clean_frame: np.ndarray, run_inference: bool = True) -> np.ndarray:
        """Run sampled inference while drawing cached boxes on intervening frames."""
        frame = cv2.resize(clean_frame, DISPLAY_SIZE)
        clean_frame_resized = frame.copy()
        if run_inference:
            self._run_detection(frame)

        for box, track_id, class_name in self._cached_detections:
            x1, y1, x2, y2 = box
            if cv2.pointPolygonTest(self.area, (x2, y2), False) < 0:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            identifier = f"ID: {track_id}" if track_id >= 0 else "Tracking..."
            cvzone.putTextRect(frame, identifier, (x2, y2), 1, 1)
            cvzone.putTextRect(frame, class_name, (x1, y1), 1, 1)
            if run_inference:
                self.crop_and_process(clean_frame_resized, box, track_id, class_name)

        cvzone.putTextRect(frame, f"Country: {self.detected_country}", (10, 30), 1, 2)
        cv2.polylines(frame, [self.area], True, (0, 255, 0), 2)
        return frame

    def start_processing(self) -> None:
        logging.info("Starting CPU video processing...")
        try:
            ret, first_frame = self.cap.read()
            if not ret:
                raise RuntimeError("The input video contains no readable frames")

            country = self.detect_country_from_scene(first_frame)
            self.detected_country = country.country
            self.country_confidence = country.confidence
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            playback_stride = max(1, math.ceil(source_fps / MAX_DISPLAY_FPS))
            display_fps = source_fps / playback_stride
            frame_period = 1.0 / display_fps
            display_frame_index = 0

            while self.cap.isOpened():
                started_at = time.perf_counter()
                ret, frame = self.cap.read()
                if not ret:
                    break

                annotated_frame = self.process_video_frame(
                    frame,
                    run_inference=display_frame_index % FRAME_STRIDE == 0,
                )
                cv2.imshow("Smart Vehicle Detector", annotated_frame)

                elapsed = time.perf_counter() - started_at
                delay_ms = max(1, round((frame_period - elapsed) * 1000))
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break

                for _ in range(playback_stride - 1):
                    if not self.cap.grab():
                        break
                display_frame_index += 1
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            logging.info("Waiting for pending Gemini vehicle analyses...")
            self._analysis_pool.shutdown(wait=True)
            with self._json_lock:
                self._write_json(self.vehicle_data_list)
            self.gemini_client.close()
            logging.info("Data saved to %s", self.output_json_path)


if __name__ == "__main__":
    SmartVehicleDetector().start_processing()
