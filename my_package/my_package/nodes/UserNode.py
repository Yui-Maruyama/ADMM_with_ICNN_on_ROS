import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_msgs.msg import Int32
from std_msgs.msg import String
from std_msgs.msg import Bool
import threading
import time
from my_package.PICNN import myPICNN
import random
import torch
import torch.optim as optim
from my_package.ADMM_utils import *
import os

class UserNode(Node):
    def __init__(self):
        super().__init__('ikisugi')
        # self.user_id = user_id
        # self.neighbor = user_list
        # self.bandwidth = bandwidth

        # self.topic_name = f'user_{self.user_id}/param'
        # self.publisher = self.create_publisher(Float32, self.topic_name, 10)
        # self.get_logger().info(f'Publishing to: {self.topic_name}')

        # self.user_id = self.declare_parameter('user_id', 0).value
        # self.bandwidth = self.declare_parameter('bandwidth', {}).value
        # self.neighbors = self.declare_parameter('neighbors', []).value
        # self.total_users = self.declare_parameter('total_users', []).value
        # self.scene = self.declare_parameter('scene', 0).value

         # Declare parameters first (optional but recommended)
         # どうやら空の配列だとbyte型と誤認されるらしい
        self.declare_parameter('neighbors', [0])
        self.declare_parameter('bandwidth', [0.0])
        self.declare_parameter('total_users', 0)
        self.declare_parameter('scene', 0)
        self.declare_parameter('user_id', -1)
        self.declare_parameter('start_time', 0.0)

        # Retrieve the parameters
        self.neighbors = self.get_parameter('neighbors').get_parameter_value().integer_array_value
        self.bandwidth = self.get_parameter('bandwidth').get_parameter_value().double_array_value
        self.bandwidth = torch.tensor(self.bandwidth)
        self.total_users = self.get_parameter('total_users').get_parameter_value().integer_value
        self.scene = self.get_parameter('scene').get_parameter_value().integer_value
        self.user_id = self.get_parameter('user_id').get_parameter_value().integer_value
        self.start_time = self.get_parameter('start_time').get_parameter_value().double_value

        self.get_logger().info(
            f"Started user {self.user_id} with neighbors={self.neighbors}, "
            f"bandwidth={self.bandwidth}, scene={self.scene}, total_users={self.total_users}")


        # self.get_logger().info(f'User ID: {self.user_id}')

        self.topic_name = f'user_{self.user_id}/param'
        self.publisher = self.create_publisher(Float32, self.topic_name, 10)
        self.finished_pub = self.create_publisher(Int32, '/finished_users', 10)  #終了時通知用
        # self.get_logger().info(f'Publishing to: {self.topic_name}')

        # 他のユーザのトピックをサブスクライブ
        for other_user_id in self.neighbors:
            if other_user_id != self.user_id:
                topic = f'user_{other_user_id}/param'
                self.create_subscription(Float32, topic, self.create_callback(other_user_id), 10)
                # self.get_logger().info(f'Subscribed to: {topic}')

        # シーンとモデルの初期化
        # self.scene = scene
        match self.scene:
            case 1:
                self.model = myPICNN(22, 10, 100)
                self.model.load_state_dict(torch.load('../model/model_scene_1.pth'))
                # self.model.load_state_dict(torch.load('../model/model_scene_1_quantized.pth'))
                # self.model = torch.jit.load('../model/model_scene_1_quantized.pth')

            case 2:
                self.model = myPICNN(16, 10, 100)
                self.model.load_state_dict(torch.load('../model/model_scene_2.pth'))
                # self.model.load_state_dict(torch.load('../model/model_scene_2_quantized.pth'))
                # self.model = torch.jit.load('../model/model_scene_2_quantized.pth')
            case 3:
                self.model = myPICNN(30, 10, 100)
                self.model.load_state_dict(torch.load('../model/model_scene_3.pth'))
                # self.model.load_state_dict(torch.load('../model/model_scene_3_quantized.pth'))
                # self.model = torch.jit.load('../model/model_scene_3_quantized.pth')

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = False

        #オブジェクトの容量読み込み
        self.max_capacities = []
        print(os.getcwd())
        capacity_info_file = "../ADMM_ROS/max_capacity.txt"
        with open(capacity_info_file, 'r') as file:
            for line in file:
                self.numbers = list(map(float, line.split()))
                self.max_capacities.append(torch.tensor(self.numbers, requires_grad=False))
        self.max_capacity = self.max_capacities[self.scene - 1]

        # 変数の初期化
        self.x = torch.tensor(np.zeros(len(self.max_capacity) * 2), requires_grad=True)
        self.mu = {user_id: torch.tensor([0.0]) for user_id in self.neighbors}
        self.s = {user_id: torch.tensor([0.0]) for user_id in self.neighbors}
        self.rho = 1.0
        self.x_list = []
        self.mu_list = []
        self.s_list = []
        self.scores = []
        self.step = 0

        # 他ユーザの通信路使用量
        # 要素数1のテンソル（初期値は例として0.0）を作成し、ユーザIDをキーとする辞書を生成
        self.other_usages = {user_id: torch.tensor([0.0]) for user_id in self.neighbors}

        # オプティマイザ
        self.optimizer = optim.Adam([self.x], lr=0.01)

        # 結果記載用ファイルの初期化
        self.result_file_name = f"../result_user_{self.total_users}/user_{self.user_id}.txt"
        with open(self.result_file_name, mode = 'w') as f:
            f.write(f"user {self.user_id}   scene:{self.scene}\n")

        # スタートを同期させる部分
        # self.optimization_started = False
        # self.optimization_thread = None
        # フラグで /start を待機
        self.start_event = threading.Event()

        # startトピックのサブスクライバ
        self.subscription = self.create_subscription(
            Bool,
            '/start',
            self.start_callback,
            10
        )

        # coordinator に準備完了を通知
        ready_msg = Int32()
        ready_msg.data = self.user_id
        self.ready_pub = self.create_publisher(Int32, '/ready', 10)
        self.ready_pub.publish(ready_msg)
        self.get_logger().info(f'User {self.user_id} is ready.')

        # 最適化スレッド開始（中で .wait() でブロック）
        self.optimization_thread = threading.Thread(target=self.optimization_loop)
        self.optimization_thread.start()

        # 終了条件の通知用
        self._finished = False
        # self.finished_pub = self.create_publisher(Int32, "finished_users", 10)
        # 毎秒終了判定を行うタイマー（メインスレッド側）
        self.create_timer(1.0, self._check_shutdown)



    def create_callback(self, other_user_id):
        def callback(msg):
            if self.user_id == 100:
                self.get_logger().info(f'Received from user_{other_user_id}: {msg.data}')
        return callback
    
    
    # 最適化を開始するための関数
    def start_callback(self, msg: Bool):
        self.get_logger().info(f'Start signal received ({msg.data}). Beginning optimization...')
        if msg.data:
            # self.get_logger().info('Start signal received. Beginning optimization...')
            self.start_event.set()


    def optimization_loop(self):
        self.get_logger().info('Waiting for /start...')
        self.start_event.wait()  # ブロックして待つ
        self.get_logger().info('Start confirmed. Running optimization.')

        while rclpy.ok() and (not is_converged(self.scores, self.step)):
            # TODO: 最適化処理をここに記述
            # xの更新
            # print(f"bandwidth constraints for user {self.user_id}: {self.bandwidth}")
            # x = local_optimize(self.user_id, self.x, self.other_usages, self.mu, self.s, self.rho, self.optimizer, self.model, self.neighbors, self.max_capacity, self.bandwidth)
            x_initial_guess = self.x.clone().detach().requires_grad_(True)
            x = local_optimize_newton(self.user_id, self.x, self.other_usages, self.mu, self.s, self.rho, self.model, self.neighbors, self.max_capacity, self.bandwidth)
            self.x = x.clone().requires_grad_()
            # self.optimizer = optim.Adam([self.x], lr=0.01)

            # sの更新
            self.s = update_s(self.user_id, self.s, self.mu, self.x, self.other_usages, self.neighbors, self.max_capacity, self.scene, self.rho, self.bandwidth)

            # muの更新
            self.mu = update_mu(self.user_id, self.s, self.x, self.mu, self.max_capacity, self.neighbors, self.scene, self.other_usages, self.bandwidth)

            # スコアの更新
            self.scores, self.x_list, self.mu_list, self.s_list = update_score(self.model, self.x, self.mu, self.s, self.x_list, self.mu_list, self.s_list, self.scores, self.result_file_name, time.time() - self.start_time, ctx = torch.tensor(1))

            # パラメータをトピックに配信
            publish_usage(self, self.x, self.max_capacity, self.other_usages, self.publisher)

            # 今は単にスリープ
            # time.sleep(1.0)

            self.step += 1
        
        # msg = Int32()
        # msg.data = self.user_id
        # self.finished_pub.publish(msg)
        self._finished = True  # 🔸 終了フラグを立てる
        self.get_logger().info(f"User {self.user_id} finished. Shutting down.")
        # 1s 後に安全に destroy
        # self.create_timer(1.0, self._safe_shutdown)
        # self.destroy_node()


    # 準備完了を管理ノードに通知
    def publish_ready_once(self):
        if not self.ready:
            msg = String()
            msg.data = self.user_id
            self.ready_pub.publish(msg)
            self.get_logger().info(f'{self.user_id} published ready.')
            self.ready = True

    
    def _check_shutdown(self):
        # if self._finished:
        #     self.get_logger().info("Shutting down node...")
        #     self.destroy_node()
        #     rclpy.shutdown()  # 🔸 これにより spin() が抜けてプログラム終了
        if self._finished:
            # 終了を通知
            msg = Int32()
            msg.data = self.user_id
            self.finished_pub.publish(msg)
            self.get_logger().info(f"Published finished signal for user {self.user_id}.")
            
            # ... 自身のノードを終了させる処理 ...
            self.destroy_node()
            rclpy.shutdown()
