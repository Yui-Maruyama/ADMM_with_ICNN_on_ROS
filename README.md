# ADMM_with_ICNN_on_ROS
ADMMを使用してICNNでモデリングされるスコアの最適化を行うためのプログラムをROS上で実装

## 環境設定
- windows 11 + WSL 2 + Ubuntu 24.04
- python: 3.12（おそらくバージョンは何でもいい）
- 必要なパッケージ: setuptools >= 65.5.0（おそらく）, colcon-common-extensions, torch, numpy, rclpy
## 実行方法
ROS 2のインストールは終わっているものとする（Linuxなら[ここ](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)を参考にすれば比較的簡単）
- 何らかの仮想環境を作成して（ROS 2インストール時と同じpythonのバージョンでないといけない？，ここでは3.12），そこに入る
```bash
python3.12 -m venv ros2_jazzy_py312
source ros2_jazzy_py312/bin/activate
source /opt/ros/jazzy/setup.bash
```

- パッケージのインストール
```bash
pip install -U pip setuptools rclpy colcon-common-extensions torch numpy
```

- 環境変数の設定
```bash
export PYTHONPATH=$PYTHONPATH:/opt/ros/jazzy/lib/python3.12/site-packages
export COLCON_PYTHON_EXECUTABLE=../ros2_jazzy_py312/bin/python3.12
```

- my_packageディレクトリ内で以下を実行
```bash
colcon build
source install/setup.bash
```

- 以下を実行
```bash
ros2 launch my_package launcher.py
```

ユーザ数などの条件を書き換えた場合は以下を再実行
```bash
rm -rf build/ install/ log/
colcon build
```
