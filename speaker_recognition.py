#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kamera-basierte Sprechererkennung
Kombiniert ReSpeaker DoA mit Computer Vision
"""

import cv2
import time
from typing import Optional
import config
from respeaker_tuning import find_respeaker
from camera_utils import CameraManager, angular_distance, draw_direction_arrow
from face_detector import FaceDetector, DetectedPerson
from camera_tracker import CameraTracker

if config.USE_OAK_D:
    from oak_camera_manager import OAKDCameraManager


class SpeakerRecognitionSystem:
    """Hauptsystem für Sprecher-Identifikation"""
    
    def __init__(self):
        # Kamera initialisieren (OAK-D oder Standard)
        if config.USE_OAK_D:
            self.camera = OAKDCameraManager()
            self.use_oak_d = True
        else:
            self.camera = CameraManager()
            self.use_oak_d = False
        
        self.face_detector = FaceDetector()
        self.camera_tracker = CameraTracker()
        self.respeaker = None
        self.running = False
        
        # Voice Activity Smoothing
        self.voice_history = []
        self.voice_smoothing_frames = config.VOICE_ACTIVITY_SMOOTHING
    
    def initialize(self) -> bool:
        """
        Initialisiert alle Komponenten
        Returns: True wenn erfolgreich
        """
        camera_type = "OAK-D Lite" if self.use_oak_d else "Webcam"
        print(f"🚀 Initialisierung Sprecher-Erkennungs-System ({camera_type})...")
        print("=" * 60)
        
        # ReSpeaker initialisieren
        print("🎤 Suche ReSpeaker USB Mic Array...")
        self.respeaker = find_respeaker()
        
        if self.respeaker is None:
            print("❌ ReSpeaker nicht gefunden!")
            print("   Bitte USB-Verbindung prüfen")
            return False
        
        print("✅ ReSpeaker gefunden!")
        
        # Kamera initialisieren
        print("📷 Öffne Kamera...")
        if not self.camera.open():
            print("❌ Kamera konnte nicht geöffnet werden!")
            print("   Bitte Kamera-Berechtigung prüfen")
            return False
        
        print("✅ Kamera bereit!")
        print("\n" + "=" * 60)
        print("System bereit! Drücke 'q' zum Beenden\n")
        
        return True
    
    def find_active_speaker(self, doa_angle: int, persons: list, 
                           has_voice: bool) -> Optional[int]:
        """
        Findet den aktiven Sprecher basierend auf DoA und Voice Activity
        
        Args:
            doa_angle: DoA-Winkel vom ReSpeaker
            persons: Liste erkannter Personen
            has_voice: Voice Activity Status
        
        Returns:
            Person ID des aktiven Sprechers oder None
        """
        # Voice Activity Smoothing
        self.voice_history.append(has_voice)
        if len(self.voice_history) > self.voice_smoothing_frames:
            self.voice_history.pop(0)
        
        # Nur als aktiv betrachten wenn Voice Activity in mehreren Frames
        voice_active = sum(self.voice_history) >= (self.voice_smoothing_frames // 2)
        
        if not voice_active or not persons:
            return None
        
        # Finde Person mit kleinstem Winkel-Unterschied
        best_match = None
        min_angle_diff = float('inf')
        
        for person in persons:
            angle_diff = angular_distance(doa_angle, person.angle)
            
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_match = person
        
        # Nur zurückgeben wenn innerhalb Threshold
        if min_angle_diff <= config.ANGLE_MATCH_THRESHOLD:
            return best_match.person_id
        
        return None
    
    def run(self):
        """Hauptschleife des Systems"""
        if not self.initialize():
            return
        
        self.running = True
        fps_time = time.time()
        frame_count = 0
        
        try:
            while self.running:
                # Frame von Kamera lesen
                success, frame = self.camera.read_frame()
                if not success:
                    print("⚠️  Konnte Frame nicht lesen")
                    break
                
                # DoA-Daten vom ReSpeaker
                try:
                    doa_angle = self.respeaker.direction
                    has_voice = self.respeaker.is_voice
                except Exception as e:
                    print(f"⚠️  ReSpeaker Fehler: {e}")
                    doa_angle = 0
                    has_voice = False
                
                # Personen im Frame erkennen
                if self.use_oak_d:
                    # OAK-D: Nutze Spatial Detections
                    spatial_detections = self.camera.get_spatial_detections()
                    persons = self._convert_spatial_to_persons(spatial_detections)
                else:
                    # Standard: OpenCV Face Detection
                    persons = self.face_detector.detect(frame, self.camera)
                
                # Aktiven Sprecher identifizieren
                active_speaker_id = self.find_active_speaker(doa_angle, persons, has_voice)
                
                # Kamera-Tracking aktualisieren
                if active_speaker_id is not None and config.TRACKING_ENABLED:
                    # Finde Person mit dieser ID
                    active_person = next((p for p in persons if p.person_id == active_speaker_id), None)
                    if active_person:
                        # Hole Entfernung (falls OAK-D)
                        distance = None
                        if self.use_oak_d:
                            for det in spatial_detections:
                                if det.bbox == active_person.bbox:
                                    distance = det.distance
                                    break
                        
                        self.camera_tracker.update_target(
                            active_person.bbox, 
                            frame.shape,
                            distance
                        )
                else:
                    # Kein aktiver Sprecher
                    self.camera_tracker.reset_target()
                
                # Tracking anwenden
                if config.TRACKING_ENABLED:
                    frame = self.camera_tracker.get_tracked_frame(frame)
                
                # Visualisierung
                # 1. DoA-Richtungspfeil
                frame = draw_direction_arrow(frame, doa_angle)
                
                # 2. Bounding Boxes und Labels
                frame = self.face_detector.draw_detections(frame, persons, active_speaker_id)
                
                # 3. Status-Informationen (Header)
                self._draw_header(frame, doa_angle, has_voice, len(persons), active_speaker_id)
                
                # 4. Tracking Status (falls aktiv)
                if config.TRACKING_ENABLED and self.camera_tracker.is_tracking():
                    self._draw_tracking_status(frame)
                
                # FPS berechnen
                frame_count += 1
                if frame_count >= 30:
                    fps = 30 / (time.time() - fps_time)
                    fps_time = time.time()
                    frame_count = 0
                else:
                    fps = 0
                
                if fps > 0:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Frame anzeigen
                cv2.imshow('Sprecher-Erkennung', frame)
                
                # Tastatur-Input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n🛑 Beende System...")
                    break
                
        except KeyboardInterrupt:
            print("\n🛑 Beende System (Strg+C)...")
        
        except Exception as e:
            print(f"\n❌ Fehler: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def _draw_header(self, frame, doa_angle: int, has_voice: bool, 
                    num_persons: int, active_speaker_id: Optional[int]):
        """Zeichnet Header mit Status-Informationen"""
        h, w = frame.shape[:2]
        
        # Header-Box
        cv2.rectangle(frame, (0, 0), (w, 60), (50, 50, 50), -1)
        
        # DoA-Winkel
        doa_text = f"🎤 DoA: {doa_angle:3d}°"
        cv2.putText(frame, doa_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Voice Activity
        voice_status = "🗣️ SPRACHE" if has_voice else "🔇 Stille"
        voice_color = (0, 255, 0) if has_voice else (150, 150, 150)
        cv2.putText(frame, voice_status, (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, voice_color, 2)
        
        # Anzahl Personen
        persons_text = f"👥 Personen: {num_persons}"
        cv2.putText(frame, persons_text, (250, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Aktiver Sprecher
        if active_speaker_id is not None:
            speaker_text = f"Aktiv: Person {active_speaker_id}"
            cv2.putText(frame, speaker_text, (250, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Kein aktiver Sprecher", (250, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    
    def _convert_spatial_to_persons(self, spatial_detections) -> list:
        """
        Konvertiert OAK-D Spatial Detections zu DetectedPerson Objekten
        
        Args:
            spatial_detections: Liste von SpatialDetection Objekten
        
        Returns:
            Liste von DetectedPerson Objekten
        """
        persons = []
        
        for i, det in enumerate(spatial_detections):
            # Berechne Winkel aus Bounding Box
            angle = self.camera.bbox_to_angle(det.bbox)
            
            # Erstelle DetectedPerson
            person = DetectedPerson(
                person_id=i,
                bbox=det.bbox,
                angle=angle,
                confidence=det.confidence
            )
            persons.append(person)
        
        return persons
    
    def _draw_tracking_status(self, frame):
        """Zeichnet Tracking-Status Overlay"""
        h, w = frame.shape[:2]
        
        status = self.camera_tracker.get_status()
        
        # Tracking Indicator (rechts oben)
        text = f"🎯 TRACKING"
        cv2.putText(frame, text, (w - 180, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        zoom_text = f"Zoom: {status['zoom']:.2f}x"
        cv2.putText(frame, zoom_text, (w - 180, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if status['distance'] is not None:
            dist_text = f"Dist: {status['distance']:.2f}m"
            cv2.putText(frame, dist_text, (w - 180, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def cleanup(self):
        """Räumt alle Ressourcen auf"""
        print("\n🧹 Räume auf...")
        
        if self.respeaker:
            self.respeaker.close()
        
        self.camera.close()
        self.face_detector.close()
        cv2.destroyAllWindows()
        
        print("✅ System beendet")


def main():
    """Haupteinstiegspunkt"""
    system = SpeakerRecognitionSystem()
    system.run()


if __name__ == "__main__":
    main()
