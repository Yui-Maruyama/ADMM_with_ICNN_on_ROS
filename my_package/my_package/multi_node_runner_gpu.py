import sys
print("Python executable:", sys.executable)

try:
    import torch
    print("Torch version:", torch.__version__)
except ImportError:
    print("Torch is NOT available")
import rclpy
from rclpy.node import Node
from my_package.nodes.UserNode import UserNode
from my_package.nodes.UserNode_GPU import UserNode_GPU
import argparse

def main():

    rclpy.init()

    # node = UserNode()
    node = UserNode_GPU()
    
    rclpy.spin(node)  # 🔸これがないとトピックの受信処理が動かない！
    
    if rclpy.ok():
        rclpy.shutdown()