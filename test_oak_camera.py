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

print("\n" + "="*60)
print("DEBUG: Checking ALL pipeline methods:")
print("="*60)
all_methods = [m for m in dir(pipeline) if not m.startswith('_')]
print(f"Total pipeline methods: {len(all_methods)}")
for method in all_methods:
    print(f"  - {method}")

print("\n" + "="*60)
print("DEBUG: Looking for XLinkOut creation methods:")
print("="*60)
xlink_methods = [m for m in dir(pipeline) if 'xlink' in m.lower() or 'link' in m.lower()]
if xlink_methods:
    print("Found these link-related methods:")
    for m in xlink_methods:
        print(f"  - {m}")
else:
    print("NO XLink methods found on pipeline!")

print("\n" + "="*60)
print("Trying to create Camera...")
print("="*60)

# Try absolutely minimal Camera setup
cam = pipeline.create(dai.node.Camera)
print(f"✓ Camera node created")

print("\n" + "="*60)
print("Checking Camera outputs:")
print("="*60)
cam_outputs = [attr for attr in dir(cam) if not attr.startswith('_') and ('out' in attr.lower() or 'video' in attr.lower() or 'preview' in attr.lower())]
print(f"Camera has {len(cam_outputs)} output-related attributes:")
for attr in cam_outputs:
    print(f"  - cam.{attr}")

print("\n" + "="*60)
print("Trying XLinkOut creation methods...")
print("="*60)

xout = None

# Method 1: pipeline.createXLinkOut()
try:
    xout = pipeline.createXLinkOut()
    print("✓ Method 1 SUCCESS: pipeline.createXLinkOut()")
except AttributeError as e:
    print(f"✗ Method 1 FAILED: {e}")

# Method 2: pipeline.create(dai.node.XLinkOut)
if xout is None:
    try:
        xout = pipeline.create(dai.node.XLinkOut)
        print("✓ Method 2 SUCCESS: pipeline.create(dai.node.XLinkOut)")
    except Exception as e:
        print(f"✗ Method 2 FAILED: {e}")

# Method 3: Check if there's an XLink class at top level
if xout is None:
    try:
        if hasattr(dai, 'XLinkOut'):
            xout = dai.XLinkOut()
            pipeline.addNode(xout)
            print("✓ Method 3 SUCCESS: dai.XLinkOut() + pipeline.addNode()")
        else:
            print("✗ Method 3 FAILED: dai.XLinkOut doesn't exist")
    except Exception as e:
        print(f"✗ Method 3 FAILED: {e}")

# Method 4: pipeline.create with string?
if xout is None:
    try:
        xout = pipeline.create("XLinkOut")
        print("✓ Method 4 SUCCESS: pipeline.create('XLinkOut')")
    except Exception as e:
        print(f"✗ Method 4 FAILED: {e}")

print("\n" + "="*50)
print("Starting device...")
print("="*50)

# XLink output
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
