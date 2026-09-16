"""
Facial Landmark Region Overlay.

Maps and visualizes dense facial landmarks on specific regions:
- Left Cheek
- Right Cheek
- Forehead
- Nose

This is a standalone module that can run on a single image or a video/webcam stream.
"""

import cv2
import numpy as np
import argparse

from facial_cues.landmarks.face_mesh import FaceMeshExtractor
from facial_cues.config.schema import LandmarksConfig

class RegionOverlay:
    """
    Modular facial landmark overlay system.
    Extracts landmarks and filters them by defined facial regions.
    """

    # MediaPipe Face Mesh landmark indices for specific regions
    # (Approximate indices mapped to standard facial regions)
    REGIONS = {
        "forehead": {
            "indices": [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                397, 365, 379, 378, 400, 377, 151, 9, 8, 107, 66, 105, 63, 70, 
                109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150
            ],
            "color": (255, 0, 0)  # Blue
        },
        "nose": {
            "indices": [
                1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 129, 358, 45, 275, 220, 
                440, 115, 344, 237, 457, 44, 274, 1, 2, 98, 327
            ],
            "color": (0, 255, 0)  # Green
        },
        "left_cheek": {
            "indices": [
                117, 118, 119, 100, 101, 50, 36, 205, 206, 207, 147, 187,
                214, 212, 216, 207, 210, 211, 32, 208, 199, 200, 201, 208
            ],
            "color": (0, 165, 255)  # Orange
        },
        "right_cheek": {
            "indices": [
                346, 347, 348, 329, 330, 280, 266, 425, 426, 427, 376, 411,
                434, 432, 436, 427, 430, 431, 262, 428, 419, 420, 421, 428
            ],
            "color": (0, 255, 255)  # Yellow
        }
    }

    def __init__(self):
        config = LandmarksConfig()
        self.face_mesh_extractor = FaceMeshExtractor(config)
        self.face_mesh_extractor.initialize()

    @classmethod
    def filter_landmarks_by_region(cls, landmarks_2d: np.ndarray) -> dict:
        """
        Filters existing landmarks into regions.
        
        Args:
            landmarks_2d: Array of shape (N, 2) containing landmark coordinates.
            
        Returns:
            Dictionary mapping region names to a list of (x, y) coordinate tuples.
        """
        region_points = {region: [] for region in cls.REGIONS.keys()}
        
        for region_name, region_data in cls.REGIONS.items():
            for idx in region_data["indices"]:
                if idx < len(landmarks_2d):
                    x, y = int(landmarks_2d[idx][0]), int(landmarks_2d[idx][1])
                    region_points[region_name].append((x, y))
                    
        return region_points

    def extract_and_mask_regions(self, image: np.ndarray) -> dict:
        """
        Extracts face mesh landmarks and filters them by the defined regions.
        
        Args:
            image: BGR numpy array image.
            
        Returns:
            Dictionary mapping region names to a list of (x, y) coordinate tuples.
        """
        meshes = self.face_mesh_extractor.extract(image)
        
        region_points = {region: [] for region in self.REGIONS.keys()}
        
        if not meshes:
            return region_points
            
        # Get first face detected
        face_landmarks_2d = meshes[0].landmarks_2d
        return self.filter_landmarks_by_region(face_landmarks_2d)

    @classmethod
    def draw_region_dots(cls, image: np.ndarray, region_points: dict, dot_size: int = 2) -> np.ndarray:
        """
        Draws colored dots for each defined region over the image.
        
        Args:
            image: Original BGR image.
            region_points: Dictionary mapping region names to (x, y) lists.
            dot_size: Radius of the dots.
            
        Returns:
            Image with dots drawn.
        """
        output_image = image.copy()
        
        for region_name, points in region_points.items():
            color = cls.REGIONS.get(region_name, {}).get("color", (255, 255, 255))
            for pt in points:
                cv2.circle(output_image, pt, dot_size, color, -1)
                
        return output_image


def run_camera():
    """Run the overlay in a real-time webcam loop."""
    overlay = RegionOverlay()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Running Region Overlay. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Extract and Mask
        region_points = overlay.extract_and_mask_regions(frame)
        
        # 2. Draw Dots
        output_frame = overlay.draw_region_dots(frame, region_points)
        
        cv2.imshow("Facial Region Overlay", output_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


def run_image(image_path: str):
    """Run the overlay on a single image."""
    overlay = RegionOverlay()
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
        
    # 1. Extract and Mask
    region_points = overlay.extract_and_mask_regions(image)
    
    # 2. Draw Dots
    output_image = overlay.draw_region_dots(image, region_points)
    
    cv2.imshow("Facial Region Overlay", output_image)
    print("Press any key to close the image.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense Facial Region Overlay")
    parser.add_argument("--image", type=str, help="Path to a single image to process. If empty, uses webcam.")
    args = parser.parse_args()
    
    if args.image:
        run_image(args.image)
    else:
        run_camera()
