import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
import threading
import time
import queue
import sys

class UserNode(Node):
    def __init__(self, user_id: str, all_user_ids: list[str]):
        super().__init__(f'user_{user_id}')
        self.user_id = user_id
        self.all_user_ids = all_user_ids   #近隣ユーザのリストをテキストかなんかで渡す
        self.buffer = queue.Queue()

        # Publisher（自身のトピック）
        self.pub = self.create_publisher(String, f'user_{user_id}/param', 10)

        # Subscriber（他ユーザのトピック）
        for uid in all_user_ids:
            if uid != user_id:
                topic = f'user_{uid}/param'
                self.create_subscription(String, topic, self.receive_callback, 10)
                self.get_logger().info(f'Subscribed to {topic}')

        # スレッド起動（非同期処理）
        threading.Thread(target=self.optimization_loop, daemon=True).start()
        threading.Thread(target=self.data_processing_loop, daemon=True).start()
        threading.Thread(target=self.publisher_loop, daemon=True).start()

    def receive_callback(self, msg):
        # 受信したデータをバッファに格納（非同期処理対象）
        self.buffer.put(msg.data)

    def data_processing_loop(self):
        while rclpy.ok():
            if not self.buffer.empty():
                data = self.buffer.get()
                self.get_logger().info(f'Received param: {data}')
            time.sleep(0.1)

    def optimization_loop(self):
        while rclpy.ok():
            # ダミー最適化処理
            self.get_logger().info('Running local optimization...')
            time.sleep(2.0)

    def publisher_loop(self):
        count = 0
        while rclpy.ok():
            msg = String()
            msg.data = f'param_from_{self.user_id}_{count}'
            self.pub.publish(msg)
            self.get_logger().info(f'Published: {msg.data}')
            count += 1
            time.sleep(3.0)


def main():
    rclpy.init()

    # 引数処理
    if len(sys.argv) < 3:
        print("Usage: ros2 run your_package user_node <user_id> <comma_separated_all_ids>")
        sys.exit(1)

    user_id = sys.argv[1]
    all_user_ids = sys.argv[2].split(",")

    node = UserNode(user_id, all_user_ids)
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()