#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kamera Tracker für digitale Nachführung
Unterstützt sowohl Standard-Webcam als auch OAK-D Lite
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import config


class CameraTracker:
    """
    Digitaler Kamera-Tracker mit Smooth Transitions
    Nutzt Depth-Informationen für intelligenten Zoom (falls verfügbar)
    """
    
    def __init__(self):
        self.target_x: Optional[float] = None
        self.target_y: Optional[float] = None
        self.target_distance: Optional[float] = None
        
        self.current_x: Optional[float] = None
        self.current_y: Optional[float] = None
        self.current_zoom: float = 1.0
        
        self.frames_on_target = 0
        self.frames_without_target = 0
        
        self.smoothing = config.TRACKING_SMOOTHING
        self.min_frames = config.TRACKING_MIN_FRAMES
        self.timeout = config.TRACKING_TIMEOUT
        self.base_zoom = config.TRACKING_ZOOM_FACTOR
        self.depth_adaptive = config.TRACKING_DEPTH_ADAPTIVE
    
    def update_target(self, bbox: Tuple[int, int, int, int], 
                     frame_shape: Tuple[int, int],
                     distance: Optional[float] = None):
        """
        Aktualisiert Ziel-Position für Tracking
        
        Args:
            bbox: (x, y, w, h) Bounding Box der Zielperson
            frame_shape: (height, width) des Frames
            distance: Optionale Entfernung in Metern (von OAK-D)
        """
        h, w = frame_shape[:2]
        x, y, box_w, box_h = bbox
        
        # Normalisierte Position (0.0 - 1.0)
        self.target_x = (x + box_w / 2) / w
        self.target_y = (y + box_h / 2) / h
        self.target_distance = distance
        
        self.frames_on_target += 1
        self.frames_without_target = 0
    
    def reset_target(self):
        """Entfernt aktuelles Ziel"""
        self.frames_without_target += 1
        
        # Nach Timeout: Reset
        if self.frames_without_target > self.timeout:
            self.target_x = None
            self.target_y = None
            self.target_distance = None
            self.frames_on_target = 0
    
    def get_tracked_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Gibt verfolgten/gezoomten Frame zurück
        
        Args:
            frame: Original RGB Frame
        
        Returns:
            Getrackte Frame mit Zoom und Crop
        """
        if not config.TRACKING_ENABLED:
            return frame
        
        h, w = frame.shape[:2]
        
        # Kein Tracking wenn noch nicht genug Frames
        if self.frames_on_target < self.min_frames:
            return frame
        
        # Kein Ziel vorhanden - sanft zurück zu Normal
        if self.target_x is None:
            if self.current_x is not None:
                # Smooth Transition zurück zur Mitte
                self.current_x += (0.5 - self.current_x) * self.smoothing * 2
                self.current_y += (0.5 - self.current_y) * self.smoothing * 2
                self.current_zoom += (1.0 - self.current_zoom) * self.smoothing * 2
                
                # Fast bei 1.0? Dann abbrechen
                if abs(self.current_zoom - 1.0) < 0.01:
                    self.current_x = None
                    self.current_y = None
                    self.current_zoom = 1.0
                    return frame
            else:
                return frame
        else:
            # Initiale Position
            if self.current_x is None:
                self.current_x = self.target_x
                self.current_y = self.target_y
                self.current_zoom = 1.0
            
            # Smooth Interpolation zur Ziel-Position
            self.current_x += (self.target_x - self.current_x) * self.smoothing
            self.current_y += (self.target_y - self.current_y) * self.smoothing
            
            # Berechne Zoom basierend auf Entfernung (falls verfügbar)
            target_zoom = self._calculate_zoom()
            self.current_zoom += (target_zoom - self.current_zoom) * self.smoothing
        
        # Anwenden von Zoom & Crop
        return self._apply_digital_zoom(frame, self.current_x, self.current_y, 
                                       self.current_zoom)
    
    def _calculate_zoom(self) -> float:
        """
        Berechnet Zoom-Faktor basierend auf Entfernung
        
        Returns:
            Zoom-Faktor (1.0 = kein Zoom)
        """
        if not self.depth_adaptive or self.target_distance is None:
            return self.base_zoom
        
        # Adaptive Zoom: Näher = weniger Zoom, Ferner = mehr Zoom
        # Normalisiere Entfernung (0.5m - 5.0m)
        dist = np.clip(self.target_distance, 
                      config.TRACKING_MIN_DISTANCE,
                      config.TRACKING_MAX_DISTANCE)
        
        # Linear Mapping: 0.5m → 1.2x, 5.0m → 2.0x
        min_zoom = 1.2
        max_zoom = 2.0
        
        normalized_dist = (dist - config.TRACKING_MIN_DISTANCE) / \
                         (config.TRACKING_MAX_DISTANCE - config.TRACKING_MIN_DISTANCE)
        
        zoom = min_zoom + (max_zoom - min_zoom) * normalized_dist
        
        return zoom
    
    def _apply_digital_zoom(self, frame: np.ndarray, 
                           center_x: float, center_y: float, 
                           zoom: float) -> np.ndarray:
        """
        Wendet digitalen Zoom und Crop an
        
        Args:
            frame: Original Frame
            center_x: X-Position des Zentrums (0.0-1.0)
            center_y: Y-Position des Zentrums (0.0-1.0)
            zoom: Zoom-Faktor
        
        Returns:
            Gezoomter und zugeschnittener Frame
        """
        h, w = frame.shape[:2]
        
        # Berechne Crop-Region
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)
        
        # Crop Zentrum
        crop_center_x = int(center_x * w)
        crop_center_y = int(center_y * h)
        
        # Crop Koordinaten
        x1 = max(0, crop_center_x - crop_w // 2)
        y1 = max(0, crop_center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        # Anpassen falls am Rand
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
        
        # Croppen
        cropped = frame[y1:y2, x1:x2]
        
        # Zurück auf Original-Größe skalieren
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return zoomed
    
    def reset(self):
        """Vollständiger Reset des Trackers"""
        self.target_x = None
        self.target_y = None
        self.target_distance = None
        self.current_x = None
        self.current_y = None
        self.current_zoom = 1.0
        self.frames_on_target = 0
        self.frames_without_target = 0
    
    def is_tracking(self) -> bool:
        """
        Prüft ob aktuell getrackt wird
        
        Returns:
            True wenn aktiv am Tracken
        """
        return (self.frames_on_target >= self.min_frames and 
                self.target_x is not None)
    
    def get_status(self) -> dict:
        """
        Gibt aktuellen Status zurück (für Debugging/Visualisierung)
        
        Returns:
            Dict mit Status-Informationen
        """
        return {
            "tracking": self.is_tracking(),
            "target_x": self.target_x,
            "target_y": self.target_y,
            "distance": self.target_distance,
            "zoom": self.current_zoom,
            "frames_on_target": self.frames_on_target
        }
