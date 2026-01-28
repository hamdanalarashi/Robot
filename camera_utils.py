#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kamera-Hilfsfunktionen für Sprechererkennung
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import config


class CameraManager:
    """Verwaltet Kamera-Zugriff und Koordinaten-zu-Winkel Konvertierung"""
    
    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.fov = config.CAMERA_FOV_HORIZONTAL
        self.camera_direction = config.CAMERA_DIRECTION
        self.frame_width = config.VIDEO_WIDTH
        self.frame_height = config.VIDEO_HEIGHT
    
    def open(self) -> bool:
        """
        Öffnet die Kamera
        Returns: True wenn erfolgreich, False sonst
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            return False
        
        # Auflösung setzen
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Liest ein Frame von der Kamera
        Returns: (success, frame)
        """
        if self.cap is None:
            return False, None
        
        success, frame = self.cap.read()
        return success, frame
    
    def position_to_angle(self, x_normalized: float) -> int:
        """
        Konvertiert horizontale Position im Frame zu absolutem Winkel
        
        Args:
            x_normalized: Position im Frame (0.0 = links, 0.5 = Mitte, 1.0 = rechts)
        
        Returns:
            Absoluter Winkel in Grad (0-359)
        """
        # Berechne Winkel relativ zur Kamera-Mitte
        # x_normalized = 0.0 → links = -FOV/2
        # x_normalized = 0.5 → Mitte = 0
        # x_normalized = 1.0 → rechts = +FOV/2
        angle_from_center = (x_normalized - 0.5) * self.fov
        
        # Absoluter Winkel
        absolute_angle = (self.camera_direction + angle_from_center) % 360
        
        return int(absolute_angle)
    
    def bbox_to_angle(self, bbox: Tuple[int, int, int, int]) -> int:
        """
        Berechnet Winkel aus Bounding Box
        
        Args:
            bbox: (x, y, w, h) in Pixel-Koordinaten
        
        Returns:
            Absoluter Winkel in Grad
        """
        x, y, w, h = bbox
        
        # Mitte der Bounding Box
        center_x = x + w / 2
        
        # Normalisiere auf 0.0 - 1.0
        if self.cap is not None:
            frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            x_normalized = center_x / frame_width
        else:
            x_normalized = center_x / self.frame_width
        
        return self.position_to_angle(x_normalized)
    
    def close(self):
        """Schließt die Kamera"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def angular_distance(angle1: int, angle2: int) -> int:
    """
    Berechnet kürzeste Distanz zwischen zwei Winkeln
    
    Args:
        angle1, angle2: Winkel in Grad (0-359)
    
    Returns:
        Kürzeste Winkel-Distanz (0-180)
    """
    diff = abs(angle1 - angle2)
    if diff > 180:
        diff = 360 - diff
    return diff


def draw_direction_arrow(frame: np.ndarray, angle: int, 
                         color: Tuple[int, int, int] = config.COLOR_DOA_ARROW,
                         length: int = 100) -> np.ndarray:
    """
    Zeichnet einen Richtungspfeil im Frame
    
    Args:
        frame: Video-Frame
        angle: Richtungs-Winkel (0° = rechts, 90° = unten)
        color: Farbe des Pfeils (BGR)
        length: Länge des Pfeils in Pixeln
    
    Returns:
        Frame mit gezeichnetem Pfeil
    """
    h, w = frame.shape[:2]
    center = (w // 2, h - 50)
    
    # Konvertiere Winkel zu Radians (OpenCV: 0° = rechts, gegen Uhrzeigersinn)
    # DoA: 0° = Nord (oben), im Uhrzeigersinn
    # Umrechnung: angle_cv = 90 - angle
    angle_rad = np.radians(90 - angle)
    
    end_x = int(center[0] + length * np.cos(angle_rad))
    end_y = int(center[1] - length * np.sin(angle_rad))
    
    # Pfeil zeichnen
    cv2.arrowedLine(frame, center, (end_x, end_y), color, 3, tipLength=0.3)
    
    return frame
