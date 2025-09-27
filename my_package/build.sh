#!/bin/bash

# スクリプトの途中でエラーが発生したら、そこで処理を停止する
set -e

# --- ここから実行したいコマンドを記述 ---

echo "ROS 2ワークスペースのビルドを開始します..."

rm -rf build install log

# 1. colconビルドを実行
#    必要に応じて --symlink-install などのオプションを追加してください
colcon build

echo "ビルドが完了しました。"

# 2. 環境設定ファイルをsource（読み込み）
#    これを行わないと、ビルドしたパッケージをROS 2が見つけられない
source install/setup.bash

echo "環境設定を読み込みました。スクリプトは正常に終了しました。"

ros2 launch my_package launcher.py
