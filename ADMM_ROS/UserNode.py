import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading
import time
from PICNN import myPICNN
import random
import time
import torch
from ADMM_utils import *

class UserNode(Node):
    def __init__(self, user_id: int, user_list: list, total_users: int):
        super().__init__(f'user_{user_id}')
        self.id = user_id
        self.neighbor = user_list

        self.topic_name = f'user_{self.user_id}/param'
        self.publisher = self.create_publisher(String, self.topic_name, 10)
        self.get_logger().info(f'Publishing to: {self.topic_name}')

        # 他のユーザのトピックをサブスクライブ
        for other_user_id in self.neighbor:
            if other_user_id != self.id:
                topic = f'user_{other_user_id}/param'
                self.create_subscription(String, topic, self.create_callback(other_user_id), 10)
                self.get_logger().info(f'Subscribed to: {topic}')

        # シーンとモデルの初期化
        random.seed(1)
        self.scene = random.randint(1, 3)
        match self.scene:
            case 1:
                self.model = myPICNN(22, 10, 100)
                self.model.load_state_dict(torch.load('../model/model_scene_1.pth'))
                self.model.train()
            case 2:
                self.model = myPICNN(16, 10, 100)
                self.model.load_state_dict(torch.load('../model/model_scene_2.pth'))
                self.model.train()
            case 3:
                self.model = myPICNN(30, 10, 100)
                self.model.load_state_dict(torch.load('../model/model_scene_3.pth'))
                self.model.train()

        #オブジェクトの容量読み込み
        capacity_info_file = "./max_capacity.txt"
        with open(capacity_info_file, 'r') as file:
            for line in file:
                self.numbers = list(map(float, line.split()))
                self.max_capacities.append(torch.tensor(self.numbers, requires_grad=False))
        self.max_capacty = self.max_capacities[self.scene - 1]

        # 変数の初期化
        self.x = torch.tensor(np.zeros(len(self.max_capacty) * 2), requires_grad=True)
        self.mu = torch.tensor(np.zeros(len(user_list)))
        self.s = torch.tensor(np.zeros(len(user_list)))
        self.rho = 1.0
        self.x_list = []
        self.mu_list = []
        self.s_list = []
        self.scores = []

        # 他ユーザの通信路使用量
        self.other_usages = [0 for _ in range(len(user_list))]

        # オプティマイザ
        self.optimizer = optim.Adam(self.x, lr=0.01)

        # 結果記載用ファイルの初期化
        self.result_file_name = f"./result_user_{total_users}/user_{user_id}.txt"
        with open(self.result_file_name, mode = 'w') as f:
            f.write(f"user {user_id}   scene:{self.scene}")

        # 最適化ループ（とりあえず空）
        self.optimization_thread = threading.Thread(target=self.optimization_loop)
        self.optimization_thread.start()

        # 時間計測
        self.start_time = time.time()

    def create_callback(self, other_user_id):
        def callback(msg):
            self.get_logger().info(f'Received from user_{other_user_id}: {msg.data}')
        return callback

    def optimization_loop(self):
        while rclpy.ok():
            # TODO: 最適化処理をここに記述
            # xの更新
            self.x = local_optimize(self.user_id, self.x, self.other_usages, self.mu, self.s, self.rho, self.optimizer, self.model, self.neighbor, self.max_capacity)

            # sの更新
            self.s = update_s(self.user_id, self.s, self.mu, self.x, self.other_usages, self.neighbor, self.max_capacity, self.scene, self.rho)

            # muの更新
            self.mu = update_mu(self.user_id, self.s, self.x, self.mu, self.max_capacity, self.neighbor, self.scene, self.other_usages)

            # スコアの更新
            self.scores, self.x_list, self.mu_list, self.s_list = update_score(self.model, self.x, self.mu, self.s, self.x_list, self.mu_list, self.s_list, self.scores, self.result_file_name, time.time() - self.time, ctx = torch.tensor(1))

            # パラメータをトピックに配信
            publish_usage(self.id, self.x, self.max_capacity, self.other_usages, self.publisher)

            # 今は単にスリープ
            time.sleep(1.0)