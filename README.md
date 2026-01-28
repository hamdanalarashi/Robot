# Speaker Recognition System mit Kamera-Tracking

Automatisches Sprecher-Erkennungssystem mit Kamera-Nachführung für Raspberry Pi / Roboter.

## Features

- 🎤 **ReSpeaker Integration** - Direction of Arrival (DoA) Erkennung
- 👥 **Gesichtserkennung** - OpenCV Haar Cascades + Spatial AI
- 🎯 **Automatisches Tracking** - Kamera folgt sprechendem Person
- 📹 **OAK-D Lite Support** - Stereo-Vision mit Tiefenerkennung
- 🌊 **Smooth Transitions** - Sanfte Zoom- und Pan-Bewegungen
- 🤖 **Raspberry Pi Ready** - Optimiert für ARM-Architektur

## Hardware-Anforderungen

- Raspberry Pi 4 (empfohlen 4GB+ RAM)
- ReSpeaker USB Mic Array (für DoA)
- OAK-D Lite Kamera (optional, Webcam funktioniert auch)
- USB 3.0 Ports

## Installation auf Raspberry Pi

```bash
# Repository klonen
git clone https://github.com/DEIN-USERNAME/speaker-recognition.git
cd speaker-recognition

# Python Virtual Environment erstellen
python3 -m venv venv
source venv/bin/activate

# Dependencies installieren
pip install -r requirements.txt

# System starten
python speaker_recognition.py
```

## Konfiguration

Passe `config.py` an deine Hardware an:

```python
# OAK-D Lite nutzen (True) oder Webcam (False)
USE_OAK_D = True  

# Tracking aktivieren/deaktivieren
TRACKING_ENABLED = True
TRACKING_ZOOM_FACTOR = 1.5  # Zoom-Level

# Kamera FOV anpassen
CAMERA_FOV_HORIZONTAL = 70  # Grad
```

## Verwendung

### Mit OAK-D Lite
```bash
# In config.py: USE_OAK_D = True
python speaker_recognition.py
```

### Mit Standard-Webcam
```bash
# In config.py: USE_OAK_D = False
python speaker_recognition.py
```

### Beenden
Drücke `q` im Kamera-Fenster oder `Ctrl+C` im Terminal.

## Steuerung

- `q` - Programm beenden
- Tracking erfolgt automatisch basierend auf Voice Activity

## Architektur

```
ReSpeaker (DoA) ──┐
                  ├──> Speaker Recognition System
OAK-D/Webcam   ──┘         │
                           ├──> Camera Tracker
                           └──> Tracked Video Output
```

## Module

- `speaker_recognition.py` - Hauptsystem
- `oak_camera_manager.py` - OAK-D Lite Interface
- `camera_tracker.py` - Digitales Tracking
- `camera_utils.py` - Kamera-Utilities
- `face_detector.py` - Gesichtserkennung
- `respeaker_tuning.py` - ReSpeaker Interface
- `config.py` - Konfiguration

## Requirements

Siehe `requirements.txt` für vollständige Liste.

Hauptabhängigkeiten:
- `opencv-python` - Computer Vision
- `depthai` - OAK-D Lite SDK
- `numpy` - Numerische Operationen
- `pyusb` - USB-Kommunikation (ReSpeaker)

## Troubleshooting

### ReSpeaker nicht gefunden
```bash
# USB-Berechtigungen prüfen
lsusb | grep 2886:0018
sudo chmod 666 /dev/bus/usb/XXX/XXX
```

### OAK-D Lite nicht erkannt
```bash
# USB 3.0 Verbindung prüfen
dmesg | grep -i usb
```

### Kamera-Berechtigung
```bash
# User zur video-Gruppe hinzufügen (Raspberry Pi)
sudo usermod -a -G video $USER
```

## Lizenz

MIT License

## Autor

Hamdan Al-Arashi - Bachelorarbeit 2026
