#!/usr/bin/env python3
"""
Super simple OAK-D test using DepthAI 3.3.0 API
Based on new pipeline.add() and cam.requestOutput() methods
"""
import depthai as dai
import cv2

print(f"DepthAI Version: {dai.__version__}")

# Create pipeline
pipeline = dai.Pipeline()

# Create Camera node
cam = pipeline.create(dai.node.Camera)
print("✓ Camera created")

# Request video output from camera
# This creates the XLinkOut internally!
cam.requestOutput((640, 480), dai.ImgFrame.Type.BGR888p, "video")
print("✓ Video output requested")

print("\n" + "="*50)
print("Starting device...")
print("="*50)

try:
    # Start device
    device = dai.Device(pipeline)
    print("✓ Device started!")
    
    # Get output queue
    q = device.getOutputQueue(name="video", maxSize=4, blocking=False)
    print("✓ Output queue created")
    
    print("\nShowing camera feed - press 'q' to quit")
    print("="*50)
    
    while True:
        frame_data = q.get()
        if frame_data is None:
            continue
        
        frame = frame_data.getCvFrame()
        
        # Add info text
        cv2.putText(frame, f"OAK-D WORKS! {frame.shape[1]}x{frame.shape[0]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("OAK-D Camera", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    print("\n✓✓✓ SUCCESS! OAK-D Camera working! ✓✓✓")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    cv2.destroyAllWindows()
