from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np
import rclpy
import os

# launcherを使用，1ノード1プロセス
def generate_launch_description():
    # 仮想環境のpythonのパスに合わせて変更してください
    venv_python = os.path.expanduser('/mnt/c/research/program/ADMM_with_ICNN_on_ROS/ros2_jazzy_py312/bin/python')

    nodes = []

    num_usr = 100   #ユーザ数
    np.random.seed(1)
    scenes = [np.random.randint(1, 4) for _ in range(num_usr)]

    rclpy.init()

    # ユーザIDリスト（整数）をファイルから読み込む
    user_list = [i for i in range(num_usr)]
    neighbors = read_txt_to_2d_list(f"../neighbors_{num_usr}.txt")
    # neighbors = [[0]]
    #帯域制約
    bandwidth = [np.zeros(num_usr) for _ in range(num_usr)]
    for i in range(num_usr):
        for j in range(num_usr):
            b = np.random.randint(5000, 8000)
            if i == j:
                bandwidth[i][j] = 99999
            else:
                bandwidth[i][j] = b / 1000
                bandwidth[j][i] = b / 1000
                # print(f"bandwidth constraint with user {i} and user {j}", bandwidth[i][j])

    for i in range(num_usr):  # 100ノードを生成

        nodes.append(
            Node(
                package='my_package',
                executable='multi_node_runner',
                name=f'user_{i}_node',
                output='screen',
                prefix=venv_python,  # ← ここで仮想環境の python を指定
                parameters=[{
                    'neighbors': neighbors[i],
                    # 'neighbors': [0],
                    'bandwidth': bandwidth[i].tolist(),
                    'total_users': num_usr,
                    'scene': scenes[i],
                    'user_id': i
                }]
            )
        )

    return LaunchDescription(nodes)


def read_txt_to_2d_list(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # 行を分割し、空白を削除し、intに変換
            row = [int(value.strip()) for value in line.strip().split(',')]
            data.append(row)
    return data