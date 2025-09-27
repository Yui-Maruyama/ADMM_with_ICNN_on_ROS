import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from std_msgs.msg import Bool

class CoordinatorNode(Node):
    def __init__(self):
        super().__init__('coordinator_node')
        self.user_count = 3  # ユーザの総数（変更可）
        self.ready_users = set()
        self.finished_users = set() # ✨ 完了したユーザーを管理するセット

        self.sub_ready = self.create_subscription(Int32, '/ready', self.ready_callback, self.user_count)
        self.pub_start = self.create_publisher(Bool, '/start', self.user_count)

        # ✨ 新しく /finished_users トピックをサブスクライブ
        self.sub_finished = self.create_subscription(
            Int32,
            '/finished_users',
            self.finished_callback,
            self.user_count)

        self.get_logger().info('Coordinator node initialized. Waiting for users to be ready...')

        self.declare_parameter('total_users', 0)
        self.total_users = self.get_parameter('total_users').get_parameter_value().integer_value

        # 終了条件の通知用
        self._finished = False
        self.create_timer(1.0, self._check_shutdown)

    def ready_callback(self, msg):
        user_id = msg.data
        if user_id not in self.ready_users:
            self.ready_users.add(user_id)
            self.get_logger().info(f"Received ready from {user_id} ({len(self.ready_users)}/{self.total_users})")

        if len(self.ready_users) >= self.total_users:
            self.get_logger().info("All users are ready. Broadcasting start signal.")
            start_msg = Bool()
            start_msg.data = True
            self.pub_start.publish(start_msg)
            # self._finished = True
            # ✨ ready_callback の役目はここまでなので、サブスクライバを破棄して無駄な受信を防ぐ
            self.destroy_subscription(self.sub_ready)

    def _check_shutdown(self):
        if self._finished:
            self.get_logger().info("Shutting down node...")
            self.destroy_node()
            rclpy.shutdown()  # 🔸 これにより spin() が抜けてプログラム終了

    # ✨ 全員の処理が完了したかを確認する新しいコールバック
    def finished_callback(self, msg):
        user_id = msg.data
        if user_id not in self.finished_users:
            self.finished_users.add(user_id)
            self.get_logger().info(f"Received finished from User {user_id} ({len(self.finished_users)}/{self.total_users})")
        
        # 全員の処理が完了したら、システムをシャットダウン
        if len(self.finished_users) >= self.total_users:
            self.get_logger().info("All users have finished. Coordinator is shutting down.")
            rclpy.shutdown() # これにより main の spin() が抜ける

# def main(args=None):
#     rclpy.init(args=args)
#     node = CoordinatorNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()
