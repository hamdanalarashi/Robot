#!/usr/bin/env python3
"""
Test script to check available DepthAI nodes
"""
import depthai as dai

print(f"DepthAI Version: {dai.__version__}")
print(f"\nChecking dai.node attributes:")
print("=" * 50)

if hasattr(dai, 'node'):
    print("✓ dai.node exists")
    
    # Check for common nodes
    nodes_to_check = [
        'ColorCamera', 'MonoCamera', 'StereoDepth',
        'XLinkOut', 'XLinkIn',
        'YoloDetectionNetwork', 'NeuralNetwork',
        'SPIOut', 'Script'
    ]
    
    for node_name in nodes_to_check:
        if hasattr(dai.node, node_name):
            print(f"  ✓ dai.node.{node_name}")
        else:
            print(f"  ✗ dai.node.{node_name} NOT FOUND")
    
    print("\nAll dai.node attributes:")
    print("-" * 50)
    for attr in dir(dai.node):
        if not attr.startswith('_'):
            print(f"  - {attr}")
else:
    print("✗ dai.node does NOT exist")

print("\n" + "=" * 50)
print("Checking pipeline methods:")
print("=" * 50)

pipeline = dai.Pipeline()
pipeline_methods = [m for m in dir(pipeline) if 'create' in m.lower() and not m.startswith('_')]

for method in pipeline_methods:
    print(f"  - {method}")

print("\n" + "=" * 50)
print("Trying to create a simple pipeline...")
print("=" * 50)

try:
    # Try method 1: pipeline.create(dai.node.X)
    cam = pipeline.create(dai.node.ColorCamera)
    print("✓ Method 1 works: pipeline.create(dai.node.ColorCamera)")
except Exception as e:
    print(f"✗ Method 1 failed: {e}")

try:
    # Try method 2: pipeline.createColorCamera()
    pipeline2 = dai.Pipeline()
    cam = pipeline2.createColorCamera()
    print("✓ Method 2 works: pipeline.createColorCamera()")
except Exception as e:
    print(f"✗ Method 2 failed: {e}")

try:
    # Try method 3: direct instantiation
    pipeline3 = dai.Pipeline()
    cam = dai.ColorCamera()
    print("✓ Method 3 works: dai.ColorCamera()")
except Exception as e:
    print(f"✗ Method 3 failed: {e}")
