import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import time
from UserNode import UserNode
import numpy as np
import torch
from std_msgs.msg import Int32

class ManagerNode(Node):
    def __init__(self, num_users):
        super().__init__("manager_node")
        self.num_users = num_users
        self.finished_users = set()
        self.sub = self.create_subscription(
            Int32, "finished_users", self.finished_callback, 10
        )

    def finished_callback(self, msg):
        self.get_logger().info(f"User {msg.data} has finished.")
        self.finished_users.add(msg.data)

    def all_users_finished(self):
        return len(self.finished_users) == self.num_users

def main():
    num_usr = 10   #ユーザ数
    np.random.seed(1)
    scenes = [np.random.randint(1, 4) for _ in range(num_usr)]

    rclpy.init()

    # ユーザIDリスト（整数）をファイルから読み込む
    user_list = [i for i in range(num_usr)]
    neighbors = read_txt_to_2d_list(f"../neighbors_{num_usr}.txt")

    #帯域制約
    bandwidth = torch.tensor( [np.zeros(num_usr) for _ in range(num_usr)] )
    for i in range(num_usr):
        for j in range(num_usr):
            b = np.random.randint(5000, 8000)
            if i == j:
                bandwidth[i][j] = 99999
            else:
                bandwidth[i][j] = b
                bandwidth[j][i] = b
                # print(f"bandwidth constraint with user {i} and user {j}", bandwidth[i][j])
    
    # 各ユーザのノードを作成
    nodes = []
    for user_id in user_list:
        node = UserNode(user_id, neighbors[user_id], num_usr, bandwidth[user_id], scenes[user_id])
        nodes.append(node)

    # 終了条件の管理用ノード
    manager_node = ManagerNode(num_usr)

    # マルチスレッド実行
    executor = rclpy.executors.MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)

    # 管理用ノードも
    executor.add_node(manager_node)

    # try:
    #     executor.spin()
    # finally:
    #     for node in nodes:
    #         print(f"finish optimization for user {node.user_id}")
    #         node.destroy_node()
    #     rclpy.shutdown()

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if manager_node.all_users_finished():
                print("All users finished.")
                break
    finally:
        for node in nodes:
            if rclpy.ok() and node.context.ok():
                try:
                    node.destroy_node()
                except Exception:
                    pass
        manager_node.destroy_node()
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
