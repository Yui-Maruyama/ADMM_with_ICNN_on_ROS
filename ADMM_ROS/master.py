import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import time
from UserNode import UserNode

def main():
    num_usr = 5   #ユーザ数

    rclpy.init()

    # ユーザIDリスト（整数）をファイルから読み込む
    user_list = [i for i in range(num_usr)]
    neighbors = read_txt_to_2d_list(f"../neighbors_{num_usr}.txt")

    # 各ユーザのノードを作成
    nodes = []
    for user_id in user_list:
        node = UserNode(user_id, neighbors[user_id])
        nodes.append(node)

    # マルチスレッド実行
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)

    try:
        executor.spin()
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()


def read_txt_to_2d_list(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # 行を分割し、空白を削除し、intに変換
            row = [int(value.strip()) for value in line.strip().split(',')]
            data.append(row)
    return data


if __name__ == '__main__':
    main()
