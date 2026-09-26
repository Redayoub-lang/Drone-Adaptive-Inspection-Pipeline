"""
Autonomous Drone Adaptive Industrial Inspection Pipeline
Developer: Ayoub Lahmar (@Redayoub-lang)
Target Program: Bachelor's in Computer Science & Technology @ Jiangsu University (JSU)
"""

import cv2
import numpy as np


class AdaptiveInspectionPipeline:

  def __init__(self, blur_threshold: float = 60.0):
    self.blur_threshold = blur_threshold
    self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

  def evaluate_sharpness(self, gray_frame: np.ndarray) -> float:
    """Calculates spatial variance of 2D Laplacian operator."""
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())

  def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = self.evaluate_sharpness(gray)

    metrics = {
        "valid_frame": True,
        "sharpness_score": sharpness,
        "structures_detected": 0,
    }

    # Reject frames affected by high-velocity UAV motion blur
    if sharpness < self.blur_threshold:
      metrics["valid_frame"] = False
      cv2.putText(
          frame,
          "FRAME DROPPED: MOTION BLUR",
          (10, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 0, 255),
          2,
      )
      return frame, metrics

    # Adaptive Contrast Normalization (CLAHE)
    enhanced = self.clahe.apply(gray)

    # Edge and Structural Vector Extraction
    edges = cv2.Canny(enhanced, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60, minLineLength=50, maxLineGap=10
    )

    if lines is not None:
      metrics["structures_detected"] = len(lines)
      for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Telemetry HUD Overlay
    cv2.putText(
        frame,
        f"SHARPNESS: {sharpness:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        frame,
        f"STRUCTURES DETECTED: {metrics['structures_detected']}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )

    return frame, metrics


if __name__ == "__main__":
  pipeline = AdaptiveInspectionPipeline(blur_threshold=60.0)

  # Synthetic Validation Harness
  sharp_img = np.random.randint(0, 40, (480, 640, 3), dtype=np.uint8)
  cv2.line(sharp_img, (100, 400), (500, 100), (255, 255, 255), 4)

  # Run Processors
  _, sharp_metrics = pipeline.process_frame(sharp_img)
  blurred_img = cv2.GaussianBlur(sharp_img, (21, 21), 0)
  _, blur_metrics = pipeline.process_frame(blurred_img)

  print(
      f"[TEST 1 - SHARP] Status: {sharp_metrics['valid_frame']} | Score:"
      f" {sharp_metrics['sharpness_score']:.1f} | Lines:"
      f" {sharp_metrics['structures_detected']}"
  )
  print(
      f"[TEST 2 - BLURRED] Status: {blur_metrics['valid_frame']} | Score:"
      f" {blur_metrics['sharpness_score']:.1f} (Correctly Dropped)"
  )
