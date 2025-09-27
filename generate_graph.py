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
# -----------------------------

def plot_total_score():
    """
    複数のユーザーの結果ファイルから合計スコアを計算し、時系列グラフを生成する。
    """
    # 結果ファイルが格納されているディレクトリのパスを構築
    dir_name = f"result_user_{NUM_USERS}"
    if USE_GPU_RESULTS:
        dir_name += "_GPU"
    
    # スクリプトの場所を基準にディレクトリを指定
    # my_packageの一つ上の階層にあると仮定
    base_path = dir_name
    if not os.path.isdir(base_path):
        print(f"エラー: ディレクトリ '{base_path}' が見つかりません。")
        print("スクリプトが正しい場所にあるか、またはNUM_USERSとUSE_GPU_RESULTSの設定を確認してください。")
        return

    # 対象となる全ユーザーのファイルパスを取得
    file_paths = sorted(glob.glob(os.path.join(base_path, "user_*.txt")))
    if not file_paths:
        print(f"エラー: ディレクトリ '{base_path}' 内に結果ファイルが見つかりません。")
        return
        
    print(f"{len(file_paths)}個のファイルを検出しました: {file_paths}")

    # 各ファイルから時刻とスコアを抽出するための正規表現
    # 例: time: 33.70902419090271, score: [236.59833]
    line_pattern = re.compile(r"time: ([\d.]+), score: \[(-?[\d.]+)\]")

    # --- アルゴリズムの実装 ---

    # 1. 各ファイルを開き、ヘッダーをスキップ
    file_handlers = [open(path, 'r') for path in file_paths]
    for f in file_handlers:
        next(f)  # 1行目のヘッダーを読み飛ばす

    # 2. 各ファイルの最初のデータ行を読み込み、現在の状態を保持
    current_data = [None] * len(file_handlers)
    def read_next_line(file_index):
        line = file_handlers[file_index].readline()
        if not line:
            return None
        match = line_pattern.search(line)
        if match:
            time = float(match.group(1))
            score = float(match.group(2))
            return (time, score)
        return read_next_line(file_index) # データ行でなければ次を読む

    for i in range(len(file_handlers)):
        current_data[i] = read_next_line(i)

    # 3. 全ファイルの中で最も早い開始時刻を取得 (t=0とするため)
    initial_times = [d[0] for d in current_data if d is not None]
    if not initial_times:
        print("エラー: どのファイルからも有効なデータ行を読み込めませんでした。")
        return
        
    t0 = min(initial_times)
    print(f"基準時刻 (t=0) を {t0}s に設定します。")

    # プロット用のデータを格納するリスト
    plot_times = []
    plot_total_scores = []

    # 4. 全てのファイルが終端に達するまでループ
    while any(d is not None for d in current_data):
        # 4a. 現在の状態の中で最も時刻が小さいものを探す
        min_time = float('inf')
        min_idx = -1
        for i, data in enumerate(current_data):
            if data is not None and data[0] < min_time:
                min_time = data[0]
                min_idx = i

        # 4b. その時点での全ユーザーの合計スコアを計算
        # 各ユーザーのスコアは、そのユーザーの最新の値が使われる
        total_score = sum(d[1] for d in current_data if d is not None)

        # 4c. プロット用リストにデータを追加 (時間はt0で正規化)
        plot_times.append(min_time - t0)
        plot_total_scores.append(total_score)

        # 4d. 時刻が最小だったファイルのポインタを1行進める
        current_data[min_idx] = read_next_line(min_idx)
    
    # ファイルを閉じる
    for f in file_handlers:
        f.close()

    # --- グラフの描画 ---
    if not plot_times:
        print("プロットするデータがありませんでした。")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(plot_times, plot_total_scores, marker='.', linestyle='-', markersize=4)
    plt.title(f'Total Score of All Users Over Time (Users: {NUM_USERS})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Total Score')
    plt.grid(True)
    
    # グラフをファイルとして保存
    if USE_GPU_RESULTS:
        output_filename = f'./graph_time_score/total_score_plot_{NUM_USERS}_GPU.png'
    else:
        output_filename = f'./graph_time_score/total_score_plot{NUM_USERS}.png'
    plt.savefig(output_filename)
    print(f"\nグラフを '{output_filename}' として保存しました。")


if __name__ == '__main__':
    plot_total_score()