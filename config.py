#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Konfiguration für Camera-basierte Sprechererkennung
"""

# Kamera Einstellungen
CAMERA_INDEX = 0  # 0 = Standard Webcam (MacBook integriert)
CAMERA_FOV_HORIZONTAL = 70  # Field of View in Grad (Standard MacBook Webcam ~70°)
CAMERA_DIRECTION = 0  # Kamera-Richtung (0° = Nord, wird ggf. kalibriert)

# ReSpeaker Einstellungen
RESPEAKER_VENDOR_ID = 0x2886
RESPEAKER_PRODUCT_ID = 0x0018

# DoA-Matching Einstellungen
ANGLE_MATCH_THRESHOLD = 30  # Max. Winkel-Differenz für Sprecher-Zuordnung (Grad)
VOICE_ACTIVITY_SMOOTHING = 3  # Anzahl Frames für Voice Activity Glättung

# Visualisierung
COLOR_ACTIVE_SPEAKER = (0, 255, 0)  # Grün für aktiven Sprecher
COLOR_INACTIVE_PERSON = (255, 100, 0)  # Blau für andere Personen
COLOR_DOA_ARROW = (0, 255, 255)  # Gelb für DoA-Richtungspfeil
BBOX_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Performance
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
FPS_TARGET = 30

# === OAK-D Lite Einstellungen ===
# HINWEIS: Auf True setzen wenn OAK-D Lite verbunden ist (Raspberry Pi)
# Auf False setzen zum Testen mit Standard-Webcam (Mac)
USE_OAK_D = True  # OAK-D Lite nutzen
OAK_D_RESOLUTION = "1080p"  # "1080p", "4K", "720p"
OAK_D_FPS = 30  # Target FPS

# Depth Einstellungen
DEPTH_ENABLED = True  # Stereo Depth aktivieren
DEPTH_MEDIAN_FILTER = "KERNEL_7x7"  # "KERNEL_3x3", "KERNEL_5x5", "KERNEL_7x7"
DEPTH_CONFIDENCE_THRESHOLD = 200  # Min. Confidence (0-255)

# Spatial Detection
SPATIAL_DETECTION_MODEL = "face-detection-retail-0004"
SPATIAL_CONFIDENCE_THRESHOLD = 0.5  # Min. Confidence für Detections (0.0-1.0)
SPATIAL_CALC_ALGO = "AVERAGE"  # "AVERAGE", "MIN", "MAX"

# === Tracking Einstellungen ===
TRACKING_ENABLED = True  # Kamera-Nachführung aktivieren
TRACKING_ZOOM_FACTOR = 1.5  # Basis Zoom-Faktor (1.0 = kein Zoom)
TRACKING_DEPTH_ADAPTIVE = True  # Zoom basierend auf Entfernung anpassen
TRACKING_MIN_DISTANCE = 0.5  # Min. Entfernung in Metern
TRACKING_MAX_DISTANCE = 5.0  # Max. Entfernung in Metern
TRACKING_SMOOTHING = 0.15  # Smoothing-Faktor (0.0 = keine, 1.0 = sofort)
TRACKING_MIN_FRAMES = 5  # Min. Frames vor Tracking-Start
TRACKING_TIMEOUT = 60  # Frames ohne aktiven Sprecher bis Reset

