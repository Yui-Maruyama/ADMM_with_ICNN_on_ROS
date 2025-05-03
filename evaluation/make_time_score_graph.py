import numpy as np
import matplotlib.pyplot as plt

# ファイルを読み込んで各列をリストとして取得するコード
centralized_time = []
centralized_score = []
distributed_time = []
distributed_score = []

num_usr = 5

# ファイル名を指定
filename_centralized = f'../result/time_centralized_{num_usr}users.txt'
filename_distributed = f'../result/time_distributed_{num_usr}users.txt'

# ファイルを開いて読み込む
with open(filename_centralized, 'r') as file:
    for line in file:
        # 各行をスペースで分割し、数値に変換
        values = line.split()
        centralized_time.append(float(values[0]))  # 1列目をリストに追加
        centralized_score.append(float(values[1]))  # 2列目をリストに追加

with open(filename_distributed, 'r') as file:
    for line in file:
        # 各行をスペースで分割し、数値に変換
        values = line.split()
        distributed_time.append(float(values[0]))  # 1列目をリストに追加
        distributed_score.append(float(values[1]))  # 2列目をリストに追加

time = 5   #何秒までのデータ使う？
# np.random.seed(1)

time_distributed = 0
time_centralized = 0
times_distributed = []
times_centralized = []
scores_distributed = []
scores_centralized = []

i = 0
#モデルを送信するときの通信のオーバーヘッドを最初に追加しておく
#スループット無視でいいのか？
time_centralized += np.random.normal(100, 10) / 1000
while time_centralized < time:
    time_centralized += centralized_time[i]
    times_centralized.append(time_centralized)
    scores_centralized.append(centralized_score[i])
    i += 1

i = 0
while time_distributed < time:
    time_distributed += distributed_time[i]
    #通信によるオーバヘッド考慮するならここ
    time_distributed += np.random.normal(100, 10) / 1000
    #適当な地点にpingを飛ばして，その平均をとる
    times_distributed.append(time_distributed)
    scores_distributed.append(distributed_score[i])
    i += 1

plt.rcParams['font.family'] = 'Arial'         # フォントファミリー
plt.rcParams['font.size'] = 18                # 基本フォントサイズ
plt.rcParams['axes.titlesize'] = 24           # タイトルのフォントサイズ
plt.rcParams['axes.labelsize'] = 18          # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 14          # X軸の目盛りラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 14          # Y軸の目盛りラベルのフォントサイズ
plt.rcParams['legend.fontsize'] = 16          # 凡例のフォントサイズ

# print(times_distributed)

# plt.plot(times_centralized, scores_centralized, label='centralized')
plt.plot(times_distributed, scores_distributed, label='proposed')
plt.plot(times_centralized, scores_centralized, label='centralized')

plt.xlabel('time (s)')
plt.ylabel('score')
plt.legend()
plt.savefig(f"../graph/time_score_user{num_usr}.pdf", bbox_inches='tight')

plt.show()