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
import argparse

def main():

    rclpy.init()

    node = UserNode()
    
    rclpy.spin(node)  # 🔸これがないとトピックの受信処理が動かない！
    
    if rclpy.ok():
        rclpy.shutdown()