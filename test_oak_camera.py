#!/usr/bin/env python3
"""
Minimal OAK-D Camera Test
Tests if we can get video from OAK-D Lite with DepthAI 3.3.0
"""
import depthai as dai
import cv2

print(f"DepthAI Version: {dai.__version__}")
print("Creating minimal pipeline...")

# Create pipeline
pipeline = dai.Pipeline()

# Try absolutely minimal Camera setup
cam = pipeline.create(dai.node.Camera)
print(f"✓ Camera node created")

# XLink output
xout = pipeline.createXLinkOut()
xout.setStreamName("video")
print(f"✓ XLinkOut created")

# Try linking video output
try:
    cam.video.link(xout.input)
    print(f"✓ cam.video linked")
except Exception as e:
    print(f"✗ cam.video failed: {e}")
    # Try preview instead
    try:
        cam.preview.link(xout.input)
        print(f"✓ cam.preview linked")
    except Exception as e2:
        print(f"✗ cam.preview also failed: {e2}")
        print("Checking available outputs...")
        for attr in dir(cam):
            if not attr.startswith('_') and 'out' in attr.lower():
                print(f"  - {attr}")
        exit(1)

print("\n" + "="*50)
print("Starting device...")
print("="*50)

try:
    device = dai.Device(pipeline)
    print("✓ Device started successfully!")
    
    # Get output queue
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    print("✓ Output queue created")
    
    print("\nPress 'q' to quit")
    print("="*50)
    
    while True:
        frame_data = q.get()
        if frame_data is None:
            continue
        
        frame = frame_data.getCvFrame()
        
        # Add info text
        cv2.putText(frame, f"OAK-D Test - {frame.shape[1]}x{frame.shape[0]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("OAK-D Camera Test", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    print("\n✓ Camera test successful!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    cv2.destroyAllWindows()
    print("\nTest completed")
