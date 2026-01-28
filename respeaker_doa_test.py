#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReSpeaker DoA (Direction of Arrival) Test
Zeigt die Richtung zum Sprecher in Echtzeit an
"""

import time
from respeaker_tuning import find_respeaker

def main():
    print("🎤 ReSpeaker DoA-Test")
    print("=" * 50)
    
    # ReSpeaker suchen
    print("🔍 Suche ReSpeaker USB Mic Array...")
    respeaker = find_respeaker()
    
    if respeaker is None:
        print("❌ ReSpeaker nicht gefunden!")
        print("\nBitte überprüfen:")
        print("  1. USB-Kabel ist angeschlossen")
        print("  2. Gerät ist eingeschaltet")
        print("  3. Korrekte Vendor/Product ID (0x2886/0x0018)")
        return
    
    print("✅ ReSpeaker gefunden!\n")
    print("Sprich um das Mikrofon herum (Strg+C zum Beenden)")
    print("-" * 50)
    
    try:
        last_angle = None
        
        while True:
            try:
                # DoA-Winkel auslesen
                angle = respeaker.direction
                
                # Voice Activity Detection
                voice_active = respeaker.is_voice
                
                # Nur anzeigen, wenn sich der Winkel geändert hat oder Sprache erkannt wurde
                if angle != last_angle or voice_active:
                    status = "🗣️ SPRACHE" if voice_active else "🔇 Stille "
                    
                    # Visuelle Richtungsanzeige
                    arrow = get_direction_arrow(angle)
                    
                    print(f"{status} | Richtung: {angle:3d}° {arrow}", end="\r")
                    last_angle = angle
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\n⚠️  Fehler beim Auslesen: {e}")
                time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("🛑 Test beendet")
    
    finally:
        respeaker.close()


def get_direction_arrow(angle):
    """
    Konvertiert Winkel in einen visuellen Pfeil
    
    Args:
        angle: Winkel in Grad (0-359)
    
    Returns:
        str: Pfeil-Symbol der Richtung
    """
    # Normalisiere auf 8 Richtungen
    directions = ["→ ", "↗ ", "↑ ", "↖ ", "← ", "↙ ", "↓ ", "↘ "]
    index = int((angle + 22.5) / 45) % 8
    return directions[index]


if __name__ == "__main__":
    main()
