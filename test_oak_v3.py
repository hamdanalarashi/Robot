#!/usr/bin/env python3
"""
OAK-D Test with DepthAI 3.x CORRECT API
No explicit XLinkOut - uses automatic bridges!
"""
import depthai as dai
import cv2

print(f"DepthAI Version: {dai.__version__}")
print("\n" + "="*60)
print("Creating DepthAI 3.x Pipeline (automatic XLink)")
print("="*60)

# Create pipeline
pipeline = dai.Pipeline()

# Create Camera node (unified in 3.x)
cam = pipeline.create(dai.node.Camera)
print("✓ Camera node created")

# Start pipeline (replaces dai.Device(pipeline))
pipeline.start()
print("✓ Pipeline started")

# Request output with automatic XLink creation!
video_output = cam.requestOutput(size=(640, 480), type=dai.ImgFrame.Type.BGR888p)
print(f"✓ Video output created: {video_output}")

# Create output queue (this creates the XLink automatically!)
video_queue = video_output.createOutputQueue(maxSize=4, blocking=False)
print(f"✓ Output queue created: {video_queue}")

print("\n" + "="*60)
print("Streaming video - press 'q' to quit")
print("="*60)

try:
    while True:
        # Get frame from queue
        frame_msg = video_queue.get()
        if frame_msg is None:
            continue
        
        frame = frame_msg.getCvFrame()
        
        # Add text
        cv2.putText(frame, f"OAK-D WORKING! {frame.shape[1]}x{frame.shape[0]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "DepthAI 3.x API", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("OAK-D Camera", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    print("\n✓✓✓ SUCCESS! OAK-D working with DepthAI 3.x! ✓✓✓")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\nPipeline stopped")
