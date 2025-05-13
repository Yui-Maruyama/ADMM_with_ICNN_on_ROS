import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt
import sys
from PICNN import *

def utility(x, scenes):   #目的関数
    #xはテンソルで渡す
    output = 0
    for i in range(num_usr):
        model = models[scenes[i] - 1]   #モデルの選択
        # ctx_tensor = torch.tensor([ctx], dtype=torch.float32, requires_grad=True)
        x_tensor = x[indexes[i] : indexes[i + 1]]

        output_i = model(ctx_tensor.unsqueeze(0), x_tensor.unsqueeze(0)) * (-1.0)

        output = output + output_i

        #拡張ラグランジュ関数の項
        for j in neighbors[i]:
            output = output + mu[i][j] * (g_ij(i, j, x[indexes[i] : indexes[i + 1]], x[indexes[j] : indexes[j + 1]]) + s[i][j])
            output = output + (0.5 * rho) * ((g_ij(i, j, x[indexes[i] : indexes[i + 1]], x[indexes[j] : indexes[j + 1]]) + s[i][j]) ** 2)

    # print(output)
    return output.sum()
    #拡張ラグランジュ関数法による実装に挑戦


def g_ij(i, j, parami, paramj):
    # 定数の計算を一度だけ行う
    scaling_factor = 10.0 / 1000

    # dri, fri, drj, frj をテンソルとして取得
    dri = parami[::2]
    fri = parami[1::2]
    drj = paramj[::2].clone().detach()
    frj = paramj[1::2].clone().detach()

    # num_objects[i] と max_capacities[i] に対応する部分のテンソル計算
    val_i = (max_capacities[scenes[i] - 1][:num_objects[i]] * dri * fri).sum() * scaling_factor
    val_j = (max_capacities[scenes[j] - 1][:num_objects[j]] * drj * frj).sum() * scaling_factor

    # 合計値を計算してバンド幅を引く
    val = val_i + val_j - bandwidth[i][j]

    return val


def flatten_and_index(max_capacities):   #max_capacitiesのテンソル変換用
    flat_data = []
    indices = []
    lengths = []
    start = 0
    for row in max_capacities:
        flat_data.extend(row)
        indices.append(start)
        lengths.append(len(row))
        start += len(row)
    return (torch.tensor(flat_data, dtype=torch.float32), 
            torch.tensor(indices, dtype=torch.long), 
            torch.tensor(lengths, dtype=torch.long))


def is_converged(scores, iter, window_size=50, tol=1e-8):
    global max_not_updated_count, max_value_iter
    ret1 = False
    if(iter > window_size):
        ma_now = np.sum([scores[len(scores) - window_size : len(scores)]]) / window_size
        ma_previous = np.sum([scores[len(scores) - window_size - 1 : len(scores) - 1]]) / window_size
        print("moving average: ", ma_now)
        ret1 = abs(ma_now / ma_previous - 1) < tol and ma_now > 0
    else:
        return False
    if(scores[iter] > scores[max_value_iter]):
        max_value_iter = iter
        max_not_updated_count = 0
    else:
        max_not_updated_count += 1

    ret2 = max_not_updated_count > window_size

    return ret1 and ret2


def update_s(s, x, mu, rho, i):
    # print(s[i].items())
    updated_s = {user_id: tensor.detach().clone().requires_grad_(False) for user_id, tensor in s[i].items()}
    for j in neighbors[i]:
        # update_value = max(0, updated_s[i, j] - (g_ij(i, j, x[indexes[i] : indexes[i + 1]], x[indexes[j] : indexes[j + 1]]) + updated_s[i, j]) - (mu[i, j] / rho))
        update_value = max(torch.tensor([0]), updated_s[j] - (g_ij(i, j, x[indexes[i] : indexes[i + 1]].clone().detach(), x[indexes[j] : indexes[j + 1]].clone().detach()) + updated_s[j]) - (mu[i][j] / rho))
        updated_s[j] = update_value
    
    # print(g_ij(0, 1, x[0].clone().detach(), x[1].clone().detach()), updated_s)
        # print(update_value)
                
    return updated_s


def update_mu(s, x, mu, i):
    updated_mu = {user_id: tensor.detach().clone().requires_grad_(False) for user_id, tensor in mu[i].items()}
    for j in neighbors[i]:
        # update_value = mu[i][j] + g_ij(i, j, x[indexes[i] : indexes[i + 1]], x[indexes[j] : indexes[j + 1]]) + s[i][j]
        update_value = mu[i][j] + g_ij(i, j, x[indexes[i] : indexes[i + 1]].clone().detach(), x[indexes[j] : indexes[j + 1]].clone().detach()) + s[i][j]
        updated_mu[j] = update_value

                # print(g_ij(i, j, x[i].clone().detach(), x[j].clone().detach()) + s[i][j])

    return updated_mu
    

def update_score(iter):
    sum_score = 0
    # ctx_tensor = torch.tensor([ctx], dtype=torch.float32, requires_grad=True)
    for i in range(num_usr):
        x_tensor = x[indexes[i] : indexes[i + 1]]
        # score = models[scenes[i] - 1](ctx.unsqueeze(0).float(), x[indexes[i] : indexes[i + 1]].unsqueeze(0).float())
        score = models[scenes[i] - 1](ctx_tensor.unsqueeze(0), x_tensor.unsqueeze(0))
        score = score.detach().numpy()[0][0]
        # print(f"User {i}'s Input parameter: ", x[indexes[i] : indexes[i + 1]])
        # print(f"User {i}'s score: ", score)
        sum_score += score
        scores[i].append(score)
    x_list.append(copy.deepcopy( x.detach().numpy() ))
    # x_list.append(copy.deepcopy( [x[i].detach().numpy() for i in range(num_usr)] ))
    total_scores.append(sum_score)

    # scheduler.step(sum_score)   #スコアに基づき学習率を更新

    print(f"sum score for iter {iter}: ", sum_score)
    return sum_score


def read_txt_to_2d_list(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # 行を分割し、空白を削除し、intに変換
            row = [int(value.strip()) for value in line.strip().split(',')]
            data.append(row)
    return data
            


# num_usr = 6   #ユーザ数
num_usr = int(sys.argv[1])
print(num_usr)
np.random.seed(1)
num_object_scene = [11, 8, 15]
scenes = [np.random.randint(1, 4) for _ in range(num_usr)]
num_objects = [num_object_scene[scenes[i] - 1] for i in range(num_usr)]
indexes = [int(np.sum(num_objects[0:i] * 2)) for i in range(num_usr)] + [np.sum(num_objects) * 2]
max_capacities = []

#コンテクスト
ctx = 1
ctx_tensor = torch.tensor([ctx], dtype=torch.float32, requires_grad=True)
scores = [[] for _ in range(num_usr)]
x_list = []
total_scores = []
optimization_time = 0   #最適化のみの実行時間
optimization_times = []   #イテレーションごとの最適化の時間

# 近隣のユーザIDリスト（整数）をファイルから読み込む
user_list = [i for i in range(num_usr)]
neighbors = read_txt_to_2d_list(f"../neighbors_{num_usr}.txt")

#収束判定
max_value_iter = 0
max_not_updated_count = 0

#ラグランジュ乗数
mu = []
s = []
rho = 1
# c = 1.0 + 1e-2   #ρの係数，ステップごとにc倍

# 帯域制約
bandwidth = [np.zeros(num_usr) for _ in range(num_usr)]
for i in range(num_usr):
    for j in range(num_usr):
        b = np.random.randint(5000, 8000)
        if i == j:
            bandwidth[i][j] = 99999
        else:
            bandwidth[i][j] = b / 1000
            bandwidth[j][i] = b / 1000


#モデルの読み込み
model_1 = myPICNN(22, 10, 100)
# model_1.load_state_dict(torch.load('model_scene_1_ver3_3x10x100.pth'))
model_1.load_state_dict(torch.load('../model/model_scene_1.pth'))
model_1.train()

# # モデルの重みを固定する
# for param in model_1.parameters():
#     param.requires_grad = False

model_2 = myPICNN(16, 10, 100)
# model_2.load_state_dict(torch.load('model_scene_2_version3_3x10x100.pth'))
model_2.load_state_dict(torch.load('../model/model_scene_2.pth'))
model_2.train()

# # モデルの重みを固定する
# for param in model_2.parameters():
    # param.requires_grad = False

model_3 = myPICNN(30, 10, 100)
# model_3.load_state_dict(torch.load('model_scene_3_version3_3x10x100.pth'))
model_3.load_state_dict(torch.load('../model/model_scene_3.pth'))
model_3.train()

# # モデルの重みを固定する
# for param in model_3.parameters():
#     param.requires_grad = False

models = [model_1, model_2, model_3]

#オブジェクトの容量読み込み
capacity_info_file = "./max_capacity.txt"

with open(capacity_info_file, 'r') as file:
    for line in file:
        numbers = list(map(float, line.split()))
        max_capacities.append(torch.tensor(numbers, requires_grad=False))

# 初期入力値（最適化したい変数）
size = np.sum(num_objects) * 2
# x = [np.zeros(num_objects[i]) for i in range(num_usr)]  
x = torch.tensor(np.zeros(size), dtype=torch.float32, requires_grad=True)
# x = torch.ones(size, dtype=torch.float32, requires_grad=True)

#ラグランジュ関数のパラメータ
for i in range(num_usr):
    s.append({user_id: torch.tensor([0.0]) for user_id in user_list})
    mu.append({user_id: torch.tensor([0.0]) for user_id in user_list})

#入力をテンソルに変換
max_capacities_flat, indices, lengths = flatten_and_index(max_capacities)
bandwidth = torch.as_tensor(bandwidth, dtype=torch.float32)
# indexes = torch.as_tensor(indexes, dtype=torch.long)

# 最適化用のSGDオプティマイザを定義（xを最適化対象に含める）
# optimizer = optim.SGD([x], lr=0.01)
optimizer = optim.Adam([x], lr=0.01)

# 学習率スケジューラの定義
# scheduler = StepLR(optimizer, step_size=100, gamma=0.99)
# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10 * num_usr, threshold=1e-4/num_usr)

# 最適化ループ
num_iterations = 1000

#結果を書き込むファイル

#実行時間計測
start_time = time.time()

#--------------------Main Loop---------------------------
for i in range(num_iterations):

    optimization_start_time = time.time()

    optimizer.zero_grad()  # 勾配の初期化
    output = utility(x, scenes)  # 目的関数の評価

    loss = output   # 最大化問題なので、負の符号をつけて最小化
    # print(loss)
    loss.backward()  # 勾配計算
    # print(f"Gradient before update: {x.grad}")  # 勾配の確認

    optimizer.step()  # パラメータの更新
    # scheduler.step()

    #パラメータのクリッピング
    with torch.no_grad():
        x.clamp_(min=0.0, max=1.0)

    # optimization_end_time = time.time()
    # optimization_time += optimization_end_time - optimization_start_time

    # optimization_times.append(optimization_end_time - optimization_start_time)

    update_score(i)

    for id in range(num_usr):
        s[id] = update_s(s, x, mu, rho, id)
        mu[id] = update_mu(s, x, mu, id)

    optimization_end_time = time.time()
    optimization_time += optimization_end_time - optimization_start_time

    optimization_times.append(optimization_end_time - optimization_start_time)

    # print()

    # if is_converged(total_scores, i):
        # num_iterations = i + 1
        # break

    # print(optimization_time)

    if optimization_time > 40 * num_usr - 80:
        break
#--------------------Main Loop---------------------------

num_iterations = i + 1
print(num_usr, num_iterations)

end_time = time.time()
print(f'exection time for {iter} iteration: ', '{:.2f}'.format((end_time - start_time)))

# print(f"Optimal input: x = {x}, Maximum value: {utility(x, scenes).item()}")
print(f"Optimal input: x = {x}, Maximum value: {np.sum(scores[i][num_iterations - 1] for i in range(num_usr))}")

print(len(scores))

print("constraints", bandwidth)
print("scenes", scenes)

for i in range(size):
    y_values = [x_list[it][i] for it in range(num_iterations)]
    plt.plot(range(num_iterations), y_values, label=f'ID ({i})')

# グラフの設定
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('x list')
plt.legend()
plt.savefig(f"../graph/centralized_x_list_user_{num_usr}.pdf")
# plt.show()

for i in range(num_usr):
    plt.plot(range(num_iterations), scores[i])

# グラフの設定
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('score (centralized)') 
plt.legend()
plt.savefig(f"../graph/centralized_scores_user_{num_usr}.pdf")
# plt.show()

# 各ステップごとのスコアの合計
plt.plot(range(num_iterations), total_scores)

# グラフの設定
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('total score list (centralized)')
plt.legend()
plt.savefig(f"../graph/centralized_total_user_{num_usr}_step_{iter}.pdf")
# plt.show()

#制約を満たすか確認
# max_capacitiesを1次元テンソルと追加情報に変換
max_capacities_flat, mc_indices, mc_lengths = flatten_and_index(max_capacities)
for i in range(num_usr):
        for j in neighbors[i]:
            tmp = 0
            for k in range(num_objects[i]):
                idx = indexes[i] + 2 * k
                mc_idx = mc_indices[scenes[i] - 1] + k
                tmp = tmp + max_capacities_flat[mc_idx] * x[idx] * x[idx + 1] * 10 / 1000
            for k in range(num_objects[j]):
                idx = indexes[j] + 2 * k
                mc_idx = mc_indices[scenes[j] - 1] + k
                tmp = tmp + max_capacities_flat[mc_idx] * x[idx] * x[idx + 1] * 10 / 1000
            
            print(f"(User {i} and user {j}) max: {bandwidth[i][j]}, usage: {tmp}")

for i in range(num_usr):
    print(f"User {i}'s parameter: ", x[indexes[i] : indexes[i + 1]])

print(f"optimization execution time for {num_usr} users: ", optimization_time)


with open(f'../result/time_centralized_{num_usr}users.txt', 'w') as file:   #実行時間をファイルに書き込む
    for i in range(num_iterations):
        file.write(f"{optimization_times[i]} {total_scores[i]}\n")