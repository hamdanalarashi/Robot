#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OAK-D Lite Kamera Manager
Verwaltet DepthAI Pipeline für Stereo-Vision und Spatial AI
"""

import depthai as dai
import cv2
import numpy as np
from typing import Optional, Tuple, List
import config


class SpatialDetection:
    """Wrapper für Spatial Detection Ergebnisse"""
    
    def __init__(self, bbox: Tuple[int, int, int, int], 
                 spatial_coords: Tuple[float, float, float],
                 confidence: float,
                 tracking_id: Optional[int] = None):
        self.bbox = bbox  # (x, y, w, h)
        self.spatial_coords = spatial_coords  # (x, y, z) in Metern
        self.confidence = confidence
        self.tracking_id = tracking_id
    
    @property
    def distance(self) -> float:
        """Gibt Entfernung in Metern zurück"""
        return self.spatial_coords[2]
    
    @property
    def center_x(self) -> int:
        """Gibt X-Position der Bounding Box Mitte zurück"""
        return self.bbox[0] + self.bbox[2] // 2
    
    @property
    def center_y(self) -> int:
        """Gibt Y-Position der Bounding Box Mitte zurück"""
        return self.bbox[1] + self.bbox[3] // 2


class OAKDCameraManager:
    """Verwaltet OAK-D Lite Kamera mit DepthAI SDK"""
    
    def __init__(self):
        self.device: Optional[dai.Device] = None
        self.pipeline: Optional[dai.Pipeline] = None
        self.rgb_queue: Optional[dai.DataOutputQueue] = None
        self.depth_queue: Optional[dai.DataOutputQueue] = None
        self.detection_queue: Optional[dai.DataOutputQueue] = None
        
        self.frame_width = 0
        self.frame_height = 0
        self.fov = config.CAMERA_FOV_HORIZONTAL
        self.camera_direction = config.CAMERA_DIRECTION
    
    def _create_pipeline(self) -> dai.Pipeline:
        """
        Erstellt DepthAI Pipeline für RGB + Depth + Face Detection
        
        Returns:
            Konfigurierte Pipeline
        """
        pipeline = dai.Pipeline()
        
        # === Color Camera ===
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        
        # Resolution basierend auf Config
        resolution_map = {
            "1080p": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
            "4K": dai.ColorCameraProperties.SensorResolution.THE_4_K,
            "720p": dai.ColorCameraProperties.SensorResolution.THE_720_P,
        }
        cam_rgb.setResolution(resolution_map.get(config.OAK_D_RESOLUTION, 
                                                   resolution_map["1080p"]))
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(config.OAK_D_FPS)
        
        # === Stereo Depth ===
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setCamera("left")
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setCamera("right")
        
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align to RGB
        
        # Median Filter für Noise Reduction
        median_map = {
            "KERNEL_3x3": dai.MedianFilter.KERNEL_3x3,
            "KERNEL_5x5": dai.MedianFilter.KERNEL_5x5,
            "KERNEL_7x7": dai.MedianFilter.KERNEL_7x7,
        }
        if config.DEPTH_ENABLED:
            stereo.initialConfig.setMedianFilter(
                median_map.get(config.DEPTH_MEDIAN_FILTER, dai.MedianFilter.KERNEL_7x7)
            )
        
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        
        # === Face Detection Network ===
        detection_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        detection_nn.setBlobPath(self._get_model_path())
        detection_nn.setConfidenceThreshold(config.SPATIAL_CONFIDENCE_THRESHOLD)
        detection_nn.input.setBlocking(False)
        detection_nn.setBoundingBoxScaleFactor(0.5)
        detection_nn.setDepthLowerThreshold(100)
        detection_nn.setDepthUpperThreshold(10000)
        
        # Spatial Location Calculator Config
        spatial_config = dai.SpatialLocationCalculatorConfigData()
        spatial_config.depthThresholds.lowerThreshold = 100
        spatial_config.depthThresholds.upperThreshold = 10000
        detection_nn.initialConfig.setBoundingBoxScaleFactor(0.5)
        
        cam_rgb.preview.link(detection_nn.input)
        stereo.depth.link(detection_nn.inputDepth)
        
        # === XLink Outputs ===
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)
        
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)
        
        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("detections")
        detection_nn.out.link(xout_nn.input)
        
        return pipeline
    
    def _get_model_path(self) -> str:
        """
        Gibt Pfad zum Face Detection Modell zurück
        Fallback auf MobileNet-SSD wenn Face Model nicht verfügbar
        """
        # Hinweis: Hier müsste ein vortrainiertes Modell eingebunden werden
        # Für Production sollte man ein .blob Modell verwenden
        # Placeholder - in Realität müsste hier ein echtes Modell-Blob geladen werden
        return str(dai.OpenVINO.Blob(dai.OpenVINO.Blob.Config()))
    
    def open(self) -> bool:
        """
        Öffnet OAK-D Lite und startet Pipeline
        
        Returns:
            True wenn erfolgreich, False sonst
        """
        try:
            self.pipeline = self._create_pipeline()
            self.device = dai.Device(self.pipeline)
            
            # Output Queues
            self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            self.detection_queue = self.device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            
            # Frame Dimensionen ermitteln
            in_rgb = self.rgb_queue.get()
            if in_rgb:
                frame = in_rgb.getCvFrame()
                self.frame_height, self.frame_width = frame.shape[:2]
            
            return True
            
        except Exception as e:
            print(f"❌ OAK-D Lite Fehler: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Liest RGB Frame von OAK-D
        
        Returns:
            (success, frame)
        """
        if self.rgb_queue is None:
            return False, None
        
        try:
            in_rgb = self.rgb_queue.get()
            if in_rgb is None:
                return False, None
            
            frame = in_rgb.getCvFrame()
            return True, frame
            
        except Exception:
            return False, None
    
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """
        Holt Depth Frame (für Visualisierung)
        
        Returns:
            Depth Frame oder None
        """
        if not config.DEPTH_ENABLED or self.depth_queue is None:
            return None
        
        try:
            in_depth = self.depth_queue.get()
            if in_depth is None:
                return None
            
            depth_frame = in_depth.getFrame()
            # Normalisiere für Visualisierung
            depth_frame_norm = cv2.normalize(depth_frame, None, 0, 255, 
                                            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_frame_norm, cv2.COLORMAP_JET)
            
            return depth_colored
            
        except Exception:
            return None
    
    def get_spatial_detections(self) -> List[SpatialDetection]:
        """
        Holt Spatial Detections (Personen mit 3D-Koordinaten)
        
        Returns:
            Liste von SpatialDetection Objekten
        """
        if self.detection_queue is None:
            return []
        
        detections = []
        
        try:
            in_det = self.detection_queue.tryGet()
            if in_det is None:
                return []
            
            for detection in in_det.detections:
                # Bounding Box in Pixel-Koordinaten
                bbox = self._normalize_bbox(
                    detection.xmin, detection.ymin,
                    detection.xmax, detection.ymax
                )
                
                # Spatial Koordinaten (X, Y, Z in Metern)
                spatial_coords = (
                    detection.spatialCoordinates.x / 1000.0,  # mm zu m
                    detection.spatialCoordinates.y / 1000.0,
                    detection.spatialCoordinates.z / 1000.0
                )
                
                # Distance Filter
                distance = spatial_coords[2]
                if (config.TRACKING_MIN_DISTANCE <= distance <= config.TRACKING_MAX_DISTANCE):
                    detections.append(SpatialDetection(
                        bbox=bbox,
                        spatial_coords=spatial_coords,
                        confidence=detection.confidence
                    ))
            
            return detections
            
        except Exception as e:
            print(f"⚠️  Detection Fehler: {e}")
            return []
    
    def _normalize_bbox(self, xmin: float, ymin: float, 
                       xmax: float, ymax: float) -> Tuple[int, int, int, int]:
        """
        Konvertiert normalisierte Bbox (0-1) zu Pixel-Koordinaten
        
        Returns:
            (x, y, w, h) in Pixeln
        """
        x = int(xmin * self.frame_width)
        y = int(ymin * self.frame_height)
        w = int((xmax - xmin) * self.frame_width)
        h = int((ymax - ymin) * self.frame_height)
        
        return (x, y, w, h)
    
    def bbox_to_angle(self, bbox: Tuple[int, int, int, int]) -> int:
        """
        Berechnet Winkel aus Bounding Box (kompatibel mit alter API)
        
        Args:
            bbox: (x, y, w, h) in Pixel-Koordinaten
        
        Returns:
            Absoluter Winkel in Grad
        """
        x, y, w, h = bbox
        center_x = x + w / 2
        x_normalized = center_x / self.frame_width
        
        # Berechne Winkel relativ zur Kamera-Mitte
        angle_from_center = (x_normalized - 0.5) * self.fov
        absolute_angle = (self.camera_direction + angle_from_center) % 360
        
        return int(absolute_angle)
    
    def position_to_angle(self, x_normalized: float) -> int:
        """
        Konvertiert horizontale Position zu absolutem Winkel (kompatibel mit alter API)
        
        Args:
            x_normalized: Position im Frame (0.0-1.0)
        
        Returns:
            Absoluter Winkel in Grad
        """
        angle_from_center = (x_normalized - 0.5) * self.fov
        absolute_angle = (self.camera_direction + angle_from_center) % 360
        return int(absolute_angle)
    
    def close(self):
        """Schließt OAK-D Lite Device"""
        if self.device:
            self.device.close()
            self.device = None
            self.rgb_queue = None
            self.depth_queue = None
            self.detection_queue = None
