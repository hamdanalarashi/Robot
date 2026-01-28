#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReSpeaker USB Mic Array - Tuning Library
Basiert auf: https://github.com/respeaker/usb_4_mic_array/blob/master/tuning.py
"""

import struct
import usb.core
import usb.util

# ReSpeaker USB 4 Mic Array IDs
VENDOR_ID = 0x2886
PRODUCT_ID = 0x0018

# Wichtige Parameter: (id, offset, type, max, min, r/w, info)
PARAMETERS = {
    'DOAANGLE': (21, 0, 'int', 359, 0, 'ro', 'DOA angle. Current value.'),
    'VOICEACTIVITY': (19, 32, 'int', 1, 0, 'ro', 'VAD voice activity status.'),
    'SPEECHDETECTED': (19, 22, 'int', 1, 0, 'ro', 'Speech detection status.'),
    'AGCONOFF': (19, 0, 'int', 1, 0, 'rw', 'Automatic Gain Control.'),
}


class ReSpeakerTuning:
    """Interface zur Steuerung des ReSpeaker USB Mic Array"""
    
    TIMEOUT = 100000

    def __init__(self, dev):
        self.dev = dev

    def read(self, name):
        """Liest einen Parameter vom ReSpeaker"""
        try:
            data = PARAMETERS[name]
        except KeyError:
            raise ValueError(f"Parameter '{name}' nicht gefunden")

        param_id = data[0]
        cmd = 0x80 | data[1]
        
        if data[2] == 'int':
            cmd |= 0x40

        length = 8

        response = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmd, param_id, length, self.TIMEOUT)

        response = struct.unpack(b'ii', response.tobytes())

        if data[2] == 'int':
            result = response[0]
        else:
            result = response[0] * (2. ** response[1])

        return result

    @property
    def direction(self):
        """Gibt den aktuellen DoA-Winkel zurück (0-359°)"""
        return self.read('DOAANGLE')

    @property
    def is_voice(self):
        """Prüft, ob Sprache erkannt wurde"""
        return self.read('VOICEACTIVITY') == 1

    def close(self):
        """Schließt die USB-Verbindung"""
        usb.util.dispose_resources(self.dev)


def find_respeaker(vid=VENDOR_ID, pid=PRODUCT_ID):
    """
    Findet und initialisiert den ReSpeaker
    Returns: ReSpeakerTuning-Objekt oder None
    """
    dev = usb.core.find(idVendor=vid, idProduct=pid)
    if not dev:
        return None
    
    return ReSpeakerTuning(dev)
