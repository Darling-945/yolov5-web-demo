"""
Model inference module for YOLO object detection
Supports YOLOv5, YOLOv8, and YOLOv11 models through Ultralytics
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOInference:
    """
    YOLO Inference class to handle object detection
    """
    def __init__(self, model_path: str = 'yolo11n.pt', conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize the YOLO model

        Args:
            model_path (str): Path to YOLO model file or model name
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for non-maximum suppression
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the YOLO model"""
        try:
            # Check if model is a custom model in models directory
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            custom_model_path = os.path.join(models_dir, self.model_path)

            if os.path.exists(custom_model_path):
                # Load custom model from models directory
                logger.info(f"Loading custom model: {self.model_path}")
                self.model = YOLO(custom_model_path)
            else:
                # Load predefined model (will be downloaded if not exists)
                logger.info(f"Loading predefined model: {self.model_path}")
                self.model = YOLO(self.model_path)

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {str(e)}")
            raise

    def detect(self, image_path: str, output_path: str) -> Dict:
        """
        Detect objects in an image and save the annotated result

        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image with detections

        Returns:
            dict: Detection results including summary and details
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        try:
            # Read the input image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")

            # Get original image dimensions for reference
            img_height, img_width = img.shape[:2]

            # Perform inference
            results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)

            # Draw results on the image
            annotated_img = results[0].plot()

            # Save the output image
            cv2.imwrite(output_path, annotated_img)

            # Get detection information
            detections = []
            detection_summary = {}

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    class_name = self.model.names[class_id]

                    # Calculate normalized bounding box coordinates
                    bbox_normalized = [
                        bbox[0] / img_width,   # x1 normalized
                        bbox[1] / img_height,  # y1 normalized
                        bbox[2] / img_width,   # x2 normalized
                        bbox[3] / img_height   # y2 normalized
                    ]

                    detection_info = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': bbox.tolist(),  # Original pixel coordinates
                        'bbox_normalized': bbox_normalized,  # Normalized coordinates
                        'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # Area in pixels
                    }

                    detections.append(detection_info)

                    # Update detection summary
                    if class_name in detection_summary:
                        detection_summary[class_name] += 1
                    else:
                        detection_summary[class_name] = 1

            # Calculate summary statistics
            summary = {
                'total_detections': len(detections),
                'detection_summary': detection_summary,
                'model_used': self.model_path,
                'confidence_threshold': self.conf_threshold,
                'image_dimensions': {
                    'width': img_width,
                    'height': img_height
                }
            }

            result = {
                'summary': summary,
                'detections': detections,
                'output_image_path': output_path
            }

            logger.info(f"Detection completed. Found {len(detections)} objects.")
            return result

        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            raise

    def detect_and_return_image(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Detect objects in an image and return both the annotated image and results

        Args:
            image_path (str): Path to input image

        Returns:
            tuple: (annotated_image, detection_results)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Read the input image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Perform inference
        results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)

        # Draw results on the image
        annotated_img = results[0].plot()

        # Get detection information
        detections = []
        detection_summary = {}

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            class_name = self.model.names[class_id]

            detection_info = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence': conf,
                'bbox': bbox.tolist()  # Convert to list for JSON serialization
            }

            detections.append(detection_info)

            # Update detection summary
            if class_name in detection_summary:
                detection_summary[class_name] += 1
            else:
                detection_summary[class_name] = 1

        # Calculate summary statistics
        summary = {
            'total_detections': len(detections),
            'detection_summary': detection_summary,
            'model_used': self.model_path,
            'confidence_threshold': self.conf_threshold
        }

        result = {
            'summary': summary,
            'detections': detections
        }

        logger.info(f"Detection completed. Found {len(detections)} objects.")
        return annotated_img, result

    def change_model(self, new_model_path: str):
        """
        Change the YOLO model being used

        Args:
            new_model_path (str): Path to new YOLO model file or model name
        """
        self.model_path = new_model_path
        self.load_model()


# Global instance of the inference class
# This allows for model caching to improve performance
yolo_inference = YOLOInference(iou_threshold=0.45)


def get_available_models() -> Dict[str, List[str]]:
    """
    Get a list of all available models including custom models

    Returns:
        dict: Dictionary containing predefined models and custom models
    """
    # Predefined models
    predefined_models = [
        'yolo11n.pt',      # YOLOv11 Nano
        'yolo11s.pt',      # YOLOv11 Small
        'yolo11m.pt',      # YOLOv11 Medium
        'yolo11l.pt',      # YOLOv11 Large
        'yolo11x.pt',      # YOLOv11 Extra Large
        'yolov8n.pt',      # YOLOv8 Nano
        'yolov8s.pt',      # YOLOv8 Small
        'yolov8m.pt',      # YOLOv8 Medium
        'yolov8l.pt',      # YOLOv8 Large
        'yolov8x.pt',      # YOLOv8 Extra Large
        'yolov5n.pt',      # YOLOv5 Nano
        'yolov5s.pt',      # YOLOv5 Small
        'yolov5m.pt',      # YOLOv5 Medium
        'yolov5l.pt',      # YOLOv5 Large
        'yolov5x.pt',      # YOLOv5 Extra Large
    ]

    # Get custom models from models directory
    custom_models = []
    models_dir = os.path.join(os.path.dirname(__file__), 'models')

    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith('.pt'):
                custom_models.append(file)

    return {
        'predefined_models': predefined_models,
        'custom_models': custom_models
    }


if __name__ == "__main__":
    # Example usage
    inference = YOLOInference(model_path='yolo11n.pt', conf_threshold=0.25)

    # For testing, you would need an actual image file
    # result = inference.detect('path/to/test/image.jpg', 'path/to/output/image.jpg')
    # print(result)
    pass