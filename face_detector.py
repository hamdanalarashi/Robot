#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gesichtserkennung mit OpenCV Haar Cascades
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectedPerson:
    """Repräsentiert eine erkannte Person"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    angle: int  # Winkel in Grad (0-359)
    confidence: float  # Erkennungs-Konfidenz (0.0-1.0)
    person_id: int  # Tracking-ID


class FaceDetector:
    """OpenCV Haar Cascade-basierte Gesichtserkennung"""
    
    def __init__(self):
        # Lade Haar Cascade Classifier für Gesichtserkennung
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.person_counter = 0
        self.tracked_persons = {}  # Einfaches Tracking basierend auf Position
    
    def detect(self, frame: np.ndarray, camera_manager) -> List[DetectedPerson]:
        """
        Erkennt Gesichter im Frame
        
        Args:
            frame: BGR Video-Frame (von OpenCV)
            camera_manager: CameraManager für Winkel-Berechnung
        
        Returns:
            Liste von DetectedPerson Objekten
        """
        # Konvertiere zu Graustufen (Haar Cascades benötigen Graustufen)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Führe Face Detection aus
        # Parameters: scaleFactor, minNeighbors, minSize
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        persons = []
        
        for (x, y, w, h) in faces:
            bbox = (x, y, w, h)
            
            # Berechne Winkel dieser Person
            angle = camera_manager.bbox_to_angle(bbox)
            
            # Confidence Score (Haar Cascades geben keine echte Confidence zurück)
            # Wir nutzen die Größe als Indikator - größere Gesichter = höhere Confidence
            confidence = min(1.0, (w * h) / (frame.shape[0] * frame.shape[1] * 0.2))
            
            # Tracking-ID vergeben
            person_id = self._get_or_create_id(bbox)
            
            persons.append(DetectedPerson(
                bbox=bbox,
                angle=angle,
                confidence=confidence,
                person_id=person_id
            ))
        
        return persons
    
    def _get_or_create_id(self, bbox: Tuple[int, int, int, int]) -> int:
        """
        Einfaches Tracking: Zuordnung von IDs basierend auf Position
        
        Args:
            bbox: Bounding Box der Person
        
        Returns:
            Person ID
        """
        x, y, w, h = bbox
        center = (x + w // 2, y + h // 2)
        
        # Suche nach ähnlicher Position in bereits getrackten Personen
        min_dist = float('inf')
        matched_id = None
        
        for person_id, tracked_center in self.tracked_persons.items():
            dist = np.sqrt((center[0] - tracked_center[0])**2 + 
                          (center[1] - tracked_center[1])**2)
            
            if dist < min_dist and dist < 100:  # 100 Pixel Threshold
                min_dist = dist
                matched_id = person_id
        
        if matched_id is not None:
            # Update Position
            self.tracked_persons[matched_id] = center
            return matched_id
        else:
            # Neue Person
            self.person_counter += 1
            self.tracked_persons[self.person_counter] = center
            return self.person_counter
    
    def draw_detections(self, frame: np.ndarray, persons: List[DetectedPerson],
                       active_speaker_id: Optional[int] = None) -> np.ndarray:
        """
        Zeichnet Bounding Boxes und Labels auf Frame
        
        Args:
            frame: Video-Frame
            persons: Liste erkannter Personen
            active_speaker_id: ID des aktiven Sprechers (wenn bekannt)
        
        Returns:
            Frame mit gezeichneten Detektionen
        """
        for person in persons:
            x, y, w, h = person.bbox
            
            # Farbe je nach Status
            if active_speaker_id is not None and person.person_id == active_speaker_id:
                color = (0, 255, 0)  # Grün für aktiven Sprecher
                label = f"Person {person.person_id} (SPRICHT)"
                thickness = 3
            else:
                color = (255, 100, 0)  # Blau für andere
                label = f"Person {person.person_id}"
                thickness = 2
            
            # Bounding Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Label mit Hintergrund
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                     0.6, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Winkel anzeigen
            angle_text = f"{person.angle}°"
            cv2.putText(frame, angle_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def close(self):
        """Schließt Resources (OpenCV Haar Cascades benötigen kein explizites Close)"""
        pass

