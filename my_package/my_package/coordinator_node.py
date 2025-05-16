# coordinator_node.py

import sys
print("Python executable:", sys.executable)

try:
    import torch
    print("Torch version:", torch.__version__)
except ImportError:
    print("Torch is NOT available")

import rclpy
from my_package.nodes.CoordinatorNode import CoordinatorNode

def main():
    rclpy.init()
    node = CoordinatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
