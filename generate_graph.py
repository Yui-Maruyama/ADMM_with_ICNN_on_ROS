import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

# --- ユーザーが設定する項目 ---
# 対象のユーザー数
NUM_USERS = 3 
# GPU版の結果を使うか (True/False)
USE_GPU_RESULTS = True
# USE_GPU_RESULTS = False
# -----------------------------

def plot_total_score():
    """
    複数のユーザーの結果ファイルから合計スコアを計算し、時系列グラフを生成する。
    先に終了したユーザーのスコアは、その最後の値を保持する。
    """
    # --- (ディレクトリやファイルの検索部分は変更なし) ---
    dir_name = f"result_user_{NUM_USERS}"
    if USE_GPU_RESULTS:
        dir_name += "_GPU"
    
    base_path = dir_name
    if not os.path.isdir(base_path):
        print(f"エラー: ディレクトリ '{base_path}' が見つかりません。")
        return

    file_paths = sorted(glob.glob(os.path.join(base_path, "user_*.txt")))
    if not file_paths:
        print(f"エラー: ディレクトリ '{base_path}' 内に結果ファイルが見つかりません。")
        return
        
    print(f"{len(file_paths)}個のファイルを検出しました: {file_paths}")
    line_pattern = re.compile(r"time: ([\d.]+), score: \[(-?[\d.]+)\]")

    # --- アルゴリズムの実装 (修正箇所) ---

    file_handlers = [open(path, 'r') for path in file_paths]
    for f in file_handlers:
        next(f)

    current_data = [None] * len(file_handlers)
    def read_next_line(file_index):
        line = file_handlers[file_index].readline()
        if not line: return None
        match = line_pattern.search(line)
        if match:
            return (float(match.group(1)), float(match.group(2)))
        return read_next_line(file_index)

    # ✨ 各ユーザーの最新スコアを保持するリストを初期化
    last_known_scores = [0.0] * len(file_handlers)

    for i in range(len(file_handlers)):
        data_point = read_next_line(i)
        current_data[i] = data_point
        # ✨ 最初のスコアをlast_known_scoresに保存
        if data_point is not None:
            last_known_scores[i] = data_point[1]

    initial_times = [d[0] for d in current_data if d is not None]
    if not initial_times:
        print("エラー: どのファイルからも有効なデータ行を読み込めませんでした。")
        return
        
    t0 = min(initial_times)
    print(f"基準時刻 (t=0) を {t0}s に設定します。")

    plot_times = []
    plot_total_scores = []

    while any(d is not None for d in current_data):
        min_time = float('inf')
        min_idx = -1
        for i, data in enumerate(current_data):
            if data is not None and data[0] < min_time:
                min_time = data[0]
                min_idx = i

        # ✨ 合計スコアの計算方法を変更
        # total_score = sum(d[1] for d in current_data if d is not None) # <- 修正前
        total_score = sum(last_known_scores) # <- 修正後

        plot_times.append(min_time - t0)
        plot_total_scores.append(total_score)

        # ✨ 時刻が最小だったユーザーのスコアを更新
        new_data_point = read_next_line(min_idx)
        current_data[min_idx] = new_data_point
        if new_data_point is not None:
            # ✨ 新しいスコアがあれば、last_known_scoresを更新
            last_known_scores[min_idx] = new_data_point[1]
    
    for f in file_handlers:
        f.close()

    # --- (グラフの描画部分は変更なし) ---
    if not plot_times:
        print("プロットするデータがありませんでした。")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(plot_times, plot_total_scores, marker='.', linestyle='-', markersize=4)
    plt.title(f'Total Score of All Users Over Time (Users: {NUM_USERS})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Total Score')
    plt.grid(True)
    
    output_dir = './graph_time_score/'
    os.makedirs(output_dir, exist_ok=True)
    
    if USE_GPU_RESULTS:
        output_filename = os.path.join(output_dir, f'total_score_plot_{NUM_USERS}_GPU.png')
    else:
        output_filename = os.path.join(output_dir, f'total_score_plot_{NUM_USERS}.png')
    plt.savefig(output_filename)
    print(f"\nグラフを '{output_filename}' として保存しました。")


if __name__ == '__main__':
    plot_total_score()