import numpy as np
import matplotlib.pyplot as plt
#集中型と分散型で，収束までに必要な時間をグラフにして出力

def is_converged_ma(scores, window_size, iter):   #移動平均を元にした収束判定
    tol = 1e-4
    if(iter > window_size):
        ma_now = np.sum(scores[iter - window_size : iter]) / window_size
        ma_previous = np.sum(scores[iter - window_size - 1 : iter - 1]) / window_size
        # print("moving average: ", ma_now)
        return abs(ma_now / ma_previous - 1) < tol and ma_now > 0
    else:
        return False
    
def is_converged_change_ratio(scores, window_size, iter):
    tol = 1e-3
    if(iter > window_size):
        recent_scores = scores[iter - window_size : iter]
        oscillation_width = max(recent_scores) - min(recent_scores)
        # 振動の幅がスコアに対してtol以下なら収束
        return (oscillation_width / np.average(recent_scores)) < tol
    else:
        return False

num_usr_min = 5
num_usr_max = 30

times_distributed = [[] for _ in range(num_usr_min, num_usr_max + 1)]
scores_distributed = [[] for _ in range(num_usr_min, num_usr_max + 1)]
times_centralized = [[] for _ in range(num_usr_min, num_usr_max + 1)]
scores_centralized = [[] for _ in range(num_usr_min, num_usr_max + 1)]

for i in range(num_usr_min, num_usr_max + 1):
    filename_d = f"../result/time_distributed_{i}users.txt"
    filename_c = f"../result/time_centralized_{i}users.txt"
    # ファイルを開いて読み込む
    with open(filename_d, 'r') as file:
        for line in file:
            # 各行をスペースで分割し、数値に変換
            values = line.split()
            times_distributed[i - num_usr_min].append(float(values[0]))  # 1列目をリストに追加
            scores_distributed[i - num_usr_min].append(float(values[1]))  # 2列目をリストに追加

    with open(filename_c, 'r') as file:
        for line in file:
            # 各行をスペースで分割し、数値に変換
            values = line.split()
            times_centralized[i - num_usr_min].append(float(values[0]))  # 1列目をリストに追加
            scores_centralized[i - num_usr_min].append(float(values[1]))  # 2列目をリストに追加

window_size_centralized = 128
window_size_distributed = 32

converged_times_centralized = []
converge_time_distributed = []

#ファイルの中身を順に見ていき，収束した時間を調べる
# np.random.seed(1)   #ランダムな遅延を加える場合

#1：最大値の更新が一定ステップ間ない場合
#2：移動平均の誤差が一定以下

for i in range(num_usr_min, num_usr_max + 1):
    iter = 0
    max_not_updated_count = 0
    max_value_iter = 0
    max_value = 0
    time = 0
    iter = 0
    tol = 1e-1   #移動平均の許容誤差（連続するステップで差がこれ以下になったら収束とみなす）
    while True:
        # print(i, iter)
        time += times_centralized[i - num_usr_min][iter]
        if scores_centralized[i - num_usr_min][iter] > max_value:
            max_value = scores_centralized[i - num_usr_min][iter]
            max_value_iter = iter
        #収束判定
        if iter < window_size_centralized:
            iter += 1
            continue
        else:
            # if (max_value_iter - i) > window_size_centralized:   #一定ステップの間最高値が更新されない場合
            if (iter - max_value_iter) > window_size_centralized:   #一定ステップの間最高値が更新されない場合
                print("converged with stable maximum value")
                break
            if is_converged_ma(scores_centralized[i - num_usr_min], window_size_centralized, iter):   #移動平均の誤差が小さい場合
                print("converged with moving average")
                break
            # if is_converged_change_ratio(scores_centralized[i - num_usr_min], window_size_centralized, iter):
            #     print("converged with change ratio")
            #     break
            elif (iter + 1 == len(scores_centralized[i - num_usr_min])):   #リストの最後まで到達した場合
                max_value = 0
                print("not converged?")
                break
            else:
                iter += 1
                continue
    print(f"centralized user {i} max score: {max_value}")
    i += 1
    converged_times_centralized.append(time)

for i in range(num_usr_min, num_usr_max + 1):
    iter = 0
    max_not_updated_count = 0
    max_value_iter = 0
    max_value = 0
    time = 0
    iter = 0
    tol = 1e-1   #移動平均の許容誤差（連続するステップで差がこれ以下になったら収束とみなす）
    while True:
        # print(i, iter)
        time += times_distributed[i - num_usr_min][iter]
        time += np.random.normal(100, 10) / 1000   #ランダムな遅延
        if scores_distributed[i - num_usr_min][iter] > max_value:
            max_value = scores_distributed[i - num_usr_min][iter]
            max_value_iter = iter
        #収束判定
        if iter < window_size_distributed:
            iter += 1
            continue
        else:
            # if (max_value_iter - i) > window_size_distributed:   #一定ステップの間最高値が更新されない場合
            if (iter - max_value_iter) > window_size_distributed:   #一定ステップの間最高値が更新されない場合
                print("converged with stable maximum value")
                break
            if is_converged_ma(scores_distributed[i - num_usr_min], window_size_distributed, iter):   #移動平均の誤差が小さい場合
                print("converged with moving average")
                break
            # if is_converged_change_ratio(scores_distributed[i - num_usr_min], window_size_distributed, iter):
            #     print("converged with change ratio")
            #     break
            elif (iter + 1 == len(scores_distributed[i - num_usr_min])):
                max_value = 0
                print("not converged?")
                break
            else:
                iter += 1
                continue
    converge_time_distributed.append(time)
    print(f"distributed user {i} max score: {max_value}")
    i += 1
    
#グラフにプロットしていく
plt.rcParams['font.family'] = 'Arial'         # フォントファミリー
plt.rcParams['font.size'] = 18                # 基本フォントサイズ
plt.rcParams['axes.titlesize'] = 24           # タイトルのフォントサイズ
plt.rcParams['axes.labelsize'] = 18          # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 14          # X軸の目盛りラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 14          # Y軸の目盛りラベルのフォントサイズ
plt.rcParams['legend.fontsize'] = 16          # 凡例のフォントサイズ

# print(times_distributed)

# plt.plot(range(num_usr_min, num_usr_max + 1), converged_times_centralized, label='centralized', marker='.')
plt.plot(range(num_usr_min, num_usr_max + 1), converge_time_distributed, label='proposed', marker='.')
plt.plot(range(num_usr_min, num_usr_max + 1), converged_times_centralized, label='centralized', marker='.')

plt.xlabel('number of users')
plt.ylabel('convergence time (s)')
plt.legend()
plt.savefig(f"../graph/convergence_time_updated.pdf", bbox_inches='tight')

plt.show()