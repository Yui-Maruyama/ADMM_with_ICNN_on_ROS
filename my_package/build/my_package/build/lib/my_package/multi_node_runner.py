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
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_id', type=int, required=True)

    rclpy.init()

    node = UserNode()