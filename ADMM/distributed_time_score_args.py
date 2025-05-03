import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
import psutil
import sys

torch.autograd.set_detect_anomaly(True)


class NonNegativeLinear(nn.Module):   #重みが非負の全結合
    def __init__(self, in_features, out_features):
        super(NonNegativeLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # 重みを非負に制限
        with torch.no_grad():
            self.linear.weight.data.clamp_(min=0)
        return self.linear(x)
    

class NonNegativeOutputLinear(nn.Module):   #重みが非負の全結合
    def __init__(self, in_features, out_features):
        super(NonNegativeOutputLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # 重みを非負に制限
        return nn.functional.relu(self.linear(x))
  

class myPICNN(nn.Module):
    def __init__(self, num_input, hidden_size_u, hidden_size_z, score_scaler = None, output_size = 1):
        super(myPICNN, self).__init__()

        self.x_u1 = nn.Linear(1, hidden_size_u)
        # self.w0zu = nn.Linear(1, num_input)
        self.w0zu = NonNegativeOutputLinear(1, num_input)
        self.w0yu = nn.Linear(1, num_input)
        self.w0z = NonNegativeLinear(num_input, hidden_size_z)
        self.w0y = nn.Linear(num_input, hidden_size_z)
        self.w0u =nn.Linear(1, hidden_size_z)

        self.u1_u2 = nn.Linear(hidden_size_u, hidden_size_u)
        # self.w1zu = nn.Linear(hidden_size_u, hidden_size_z)
        self.w1zu = NonNegativeOutputLinear(hidden_size_u, hidden_size_z)
        self.w1yu = nn.Linear(hidden_size_u, num_input)
        self.w1z = NonNegativeLinear(hidden_size_z, hidden_size_z)
        self.w1y = nn.Linear(num_input, hidden_size_z)
        self.w1u =nn.Linear(hidden_size_u, hidden_size_z)

        self.u2_u3 = nn.Linear(hidden_size_u, hidden_size_u)
        # self.w2zu = nn.Linear(hidden_size_u, hidden_size_z)
        self.w2zu = NonNegativeOutputLinear(hidden_size_u, hidden_size_z)
        self.w2yu = nn.Linear(hidden_size_u, num_input)
        self.w2z = NonNegativeLinear(hidden_size_z, hidden_size_z)
        self.w2y = nn.Linear(num_input, hidden_size_z)
        self.w2u =nn.Linear(hidden_size_u, hidden_size_z)

        # self.w3zu = nn.Linear(hidden_size_u, hidden_size_z)
        # self.w3yu = nn.Linear(hidden_size_u, num_input)
        # self.w3z = nn.Linear(hidden_size_z, hidden_size_z)
        # self.w3y = nn.Linear(num_input, hidden_size_z)
        # self.w3u =nn.Linear(hidden_size_u, hidden_size_z)

        self.output = NonNegativeLinear(hidden_size_z, 1)

        # self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.activation = nn.ReLU()
        self.score_scaler = score_scaler

    def forward(self, x, y):

        # u1 = torch.relu(self.x_u1(x))
        u1 = self.activation(self.x_u1(x))
        # z1_1 = self.w0z( y * self.w0zu(x) )
        z1_2 = self.w0y( y * ( self.w0yu(x) ) )
        z1_3 = self.w0u(x)
        # z1 = torch.relu(z1_1 + z1_2 + z1_3)
        # z1 = self.activation(z1_1 + z1_2 + z1_3)
        z1 = self.activation(z1_2 + z1_3)

        # u2 = torch.relu(self.u1_u2(u1))
        u2 = self.activation(self.u1_u2(u1))
        # print(self.w1zu(u1))
        z2_1 = self.w1z( z1 * self.w1zu(u1) )
        z2_2 = self.w1y( y * ( self.w1yu(u1) ) )
        z2_3 = self.w1u(u1)
        # z2 = torch.relu(z2_1 + z2_2 + z2_3)
        z2 = self.activation(z2_1 + z2_2 + z2_3)

        # u3 = torch.relu(self.u2_u3(u2))
        u3 = self.activation(self.u2_u3(u2))
        z3_1 = self.w2z( z2 * self.w2zu(u2) )
        z3_2 = self.w2y( y * ( self.w2yu(u2) ) )
        z3_3 = self.w2u(u2)
        # z3 = torch.relu(z3_1 + z3_2 + z3_3)
        z3 = self.activation(z3_1 + z3_2 + z3_3)

        # output = self.output(z3)
        output = self.output(z2)

        if self.score_scaler is not None and not self.training:
            # 逆正規化を実行
            original_output = self.score_scaler.inverse_transform(output.detach().numpy())
            return torch.tensor(original_output)

        return output



def minimize(id):   # ユーザ1人分の最適化を行う
    global optimization_time
    for i in range(iter_inner):
        # optimization_start_time = time.time()

        val = 0
        optimizers[id].zero_grad()  # 勾配のリセット
        # for param in models[id].parameters():
        #     if param.grad is not None:
        #         param.grad.detach_()
        #         param.grad.zero_()
        
        # param = x[indexes[id] : indexes[id + 1]]  # 現在のパラメータのスライス
        param = x[id]

        val = models[scenes[id] - 1](ctx.unsqueeze(0).float(), param.unsqueeze(0).float()) * (-1.0)
        
        # 他のユーザーとの制約を追加
        for j in range(num_usr):
            if j != id:
                other_param = x[j].clone().detach()
                val = val + mu[id][j] * (g_ij(id, j, param, other_param) + s[id][j])
                val = val + (0.5 * rho) * ((g_ij(id, j, param, other_param) + s[id][j]) ** 2)

        loss = val.sum()
        
        loss.backward()

        # print(i)
        # print(f"Gradient before update: {x[id].grad}")
        # print()

        optimizers[id].step()  # パラメータの更新
        # print(val.item())
        # schedulers[id].step(val.item() * (-1.0))

        #パラメータのクリッピング
        with torch.no_grad():
            x[id].copy_(param)
            x[id].clamp_(min=0.0, max=1.0)

        # optimization_end_time = time.time()
        # optimization_time += optimization_end_time - optimization_start_time


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


def update_x(x):
    global optimization_time
    # newx = []
    # x_old = x.clone().detach()
    x_old = copy.deepcopy(x)
    x_new = []
    optimization_times = []
    for id in range(num_usr):
        optimization_start_time = time.time()

        minimize(id)

        #ユーザの最適化が全部終わるまでパラメータ更新しない
        x_new.append(x[id].clone().detach())
        x[id].data = x_old[id].clone().detach()

        optimization_end_time = time.time()
        optimization_times.append(optimization_end_time - optimization_start_time)

    for id in range(num_usr):
        # x.data[indexes[id] : indexes[id + 1]] = x_new[id][indexes[id] : indexes[id + 1]]
        x[id].data = x_new[id]

    #このイテレーションにかかった時間（ユーザごとの実行時間のうち最大のもの）
    times_for_iter.append(max(optimization_times))
    optimization_time += sum(optimization_times)
    
        
    # return newx


def update_s(s, x, mu, rho):
    updated_s = s.clone().detach()
    for i in range(num_usr):
        for j in range(num_usr):
            if i != j:
                # update_value = max(0, updated_s[i, j] - (g_ij(i, j, x[indexes[i] : indexes[i + 1]], x[indexes[j] : indexes[j + 1]]) + updated_s[i, j]) - (mu[i, j] / rho))
                update_value = max(0, updated_s[i, j] - (g_ij(i, j, x[i].clone().detach(), x[j].clone().detach()) + updated_s[i, j]) - (mu[i, j] / rho))
                updated_s[i, j] = update_value
    
    # print(g_ij(0, 1, x[0].clone().detach(), x[1].clone().detach()), updated_s)
    # print(updated_s)
                
    return updated_s


def update_mu(s, x, mu):
    updated_mu = mu.clone().detach()
    for i in range(num_usr):
        for j in range(num_usr):
            if i != j:
                # update_value = mu[i][j] + g_ij(i, j, x[indexes[i] : indexes[i + 1]], x[indexes[j] : indexes[j + 1]]) + s[i][j]
                update_value = mu[i][j] + g_ij(i, j, x[i].clone().detach(), x[j].clone().detach()) + s[i][j]
                updated_mu[i, j] = update_value

                # print(g_ij(i, j, x[i].clone().detach(), x[j].clone().detach()) + s[i][j])

    return updated_mu


def update_score():
    global scores, x_list, mu_list, s_list, x, mu, s, num_usr, indexes, scenes
    sum_score = 0
    for i in range(num_usr):
        # score = models[scenes[i] - 1](ctx.unsqueeze(0).float(), x[indexes[i] : indexes[i + 1]].unsqueeze(0).float())
        score = models[scenes[i] - 1](ctx.unsqueeze(0).float(), x[i].unsqueeze(0).float())
        score = score.detach().numpy()[0][0]
        # print(f"user {i}: ", score)
        sum_score += score
        scores[i].append(score)

        # schedulers[i].step(score)   #スコアの値によって学習率を管理

    # x_list.append(copy.deepcopy( x.detach().numpy() ))
    x_list.append(copy.deepcopy( [x[i].detach().numpy() for i in range(num_usr)] ))
    mu_list.append(copy.deepcopy(mu.detach().numpy()))
    s_list.append(copy.deepcopy(s.detach().numpy()))
    total_scores.append(sum_score)
    return sum_score


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


def print_cap_constraint():
    # max_capacities_flat, mc_indices, mc_lengths = flatten_and_index(max_capacities)
    for i in range(num_usr):
            for j in range(i + 1, num_usr):
                if i != j:
                    tmp = 0
                    # for k in range(num_objects[i]):
                    #     idx = indexes[i] + 2 * k
                    #     mc_idx = mc_indices[i] + k
                    #     tmp = tmp + max_capacities_flat[mc_idx] * x[idx] * x[idx + 1] * 10 / 1000
                    # for k in range(num_objects[j]):
                    #     idx = indexes[j] + 2 * k
                    #     mc_idx = mc_indices[j] + k
                    #     tmp = tmp + max_capacities_flat[mc_idx] * x[idx] * x[idx + 1] * 10 / 1000
                    for k in range(num_objects[i]):
                        tmp = tmp + max_capacities[scenes[i] - 1][k] * x[i][2 * k] * x[i][2 * k + 1] * 10 / 1000
                    for k in range(num_objects[j]):
                        tmp = tmp + max_capacities[scenes[j] - 1][k] * x[j][2 * k] * x[j][2 * k + 1] * 10 / 1000
                
                print(f"(User {i} and user {j}) max: {bandwidth[i][j]}, usage: {tmp}")


def is_converged(scores, iter, window_size=50, tol=1e-8):
    global max_not_updated_count, max_value_iter
    ret1 = False
    if(iter > window_size):
        ma_now = np.sum([scores[len(scores) - window_size : len(scores)]]) / window_size
        ma_previous = np.sum([scores[len(scores) - window_size - 1 : len(scores) - 1]]) / window_size
        print("moving average: ", ma_now)
        if (abs(ma_now / ma_previous - 1) < tol and ma_now > 0):
            ret1 = True
    else:
        return False
    
    if(scores[iter] > scores[max_value_iter]):
        max_value_iter = iter
        max_not_updated_count = 0
    else:
        max_not_updated_count += 1
    ret2 = max_not_updated_count > window_size

    return ret1 and ret2


#パラメータの初期化
# num_usr = 21   #ユーザ数
num_usr = int(sys.argv[1])
iter = 1000  #イテレーション
iter_inner = 5  #各ユーザの最適化のイテレーション
rho = 1.0
np.random.seed(1)   #乱数の初期化
num_object_scene = [11, 8, 15]   #シーンごとのオブジェクト数
scenes = [np.random.randint(1, 4) for _ in range(num_usr)]   #各ユーザのシーン
print(scenes)
num_objects = [num_object_scene[scenes[i] - 1] for i in range(num_usr)]   #ユーザごとのオブジェクト数
indexes = [int(np.sum(num_objects[0:i] * 2)) for i in range(num_usr)] + [np.sum(num_objects) * 2]   #各ユーザのオブジェクトのインデックスの範囲
print(indexes)
max_capacities = []   #オブジェクトの最大容量
ctx = torch.tensor([1])   #コンテクスト
optimization_time = 0   #最適化+パラメータ更新にかかった時間
times_for_iter = []   #各ステップの実行時間

#収束判定
max_value_iter = 0
max_not_updated_count = 0

#パラメータの値保存用
scores = [[] for _ in range(num_usr)]
total_scores = []
x_list = []
mu_list = []
s_list = []

#オブジェクトの容量読み込み
capacity_info_file = "./max_capacity.txt"

with open(capacity_info_file, 'r') as file:
    for line in file:
        numbers = list(map(float, line.split()))
        # max_capacities.append(numbers)
        max_capacities.append(torch.tensor(numbers, requires_grad=False))

# x = torch.zeros(np.sum(num_objects) * 2, dtype=torch.float32, requires_grad=True)   #ユーザごとのテンソルで定義
# x = [ torch.tensor(np.zeros(num_objects[i] * 2), dtype=torch.float32, requires_grad=True) for i in range(num_usr) ]
x = [ torch.tensor(np.zeros(num_objects[i] * 2), requires_grad=True) for i in range(num_usr) ]

s = torch.tensor( [np.zeros(num_usr) for _ in range(num_usr)] )
mu = torch.tensor( [np.zeros(num_usr) for _ in range(num_usr)] )

#帯域制約
bandwidth = torch.tensor( [np.zeros(num_usr) for _ in range(num_usr)] )
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

model_2 = myPICNN(16, 10, 100)
# model_2.load_state_dict(torch.load('model_scene_2_version3_3x10x100.pth'))
model_2.load_state_dict(torch.load('../model/model_scene_2.pth'))
model_2.train()

model_3 = myPICNN(30, 10, 100)
# model_3.load_state_dict(torch.load('model_scene_3_version3_3x10x100.pth'))
model_3.load_state_dict(torch.load('../model/model_scene_3.pth'))
model_3.train()

models = [model_1, model_2, model_3]

#最適化のオプティマイザ，スケジューラの定義
# optimizer = optim.Adam([x], lr=0.01)
optimizers = [optim.Adam([x[i]], lr=0.01) for i in range(num_usr)]
# schedulers = [StepLR(optimizers[i], step_size=100, gamma=0.99) for i in range(num_usr)]
# schedulers = [ReduceLROnPlateau(optimizers[i], mode='max', factor=0.5, patience=10, threshold=1e-4/num_usr) for i in range(num_usr)]

#実行時間計測
start_time = time.time()

#メインループ
for k in range(iter):
    sum_score = 0
    param = copy.deepcopy(x)

    # print(x)
    update_x(x)
    # print(x)
    # x = copy.deepcopy(newx)

    s = update_s(s, x, mu, rho)
    mu = update_mu(s, x, mu)
    sum_score = update_score()

    #収束判定
    # if is_converged(total_scores, k):
    #     iter = k + 1
    #     break

    # print("user ", i, "th ", "score for iter ", k, ": ", )
    # print("user ", i, "th ", "parameter for iter ", k, ": ", x[indexes[i] : indexes[i + 1]])
    # print(x)
    print("sum score for iter ", k, ": ", sum_score)
    # print_cap_constraint()
    # print()

    # print(optimization_time)
    if optimization_time > 500.0:
        break

iter = k + 1


#実行時間計測
end_time = time.time()
print(f'exection time for {iter} iteration, inner {iter_inner} iteration: ', '{:.2f}'.format((end_time - start_time)))

print("constraints", bandwidth)
print("scenes", scenes)

for j in range(num_usr):
    for k in range(num_usr):
        plt.plot(range(iter), np.array(mu_list)[:, j, k], label=f'ID ({j}, {k})')

# グラフの設定
# plt.xlabel('Iteration')
# plt.ylabel('Value')
# plt.title('mu list')
# plt.legend()
# plt.show()

# print(np.array(mu_list).shape)
for j in range(num_usr):
    for k in range(num_usr):
        plt.plot(range(iter), np.array(s_list)[:, j, k], label=f'ID ({j}, {k})')

# グラフの設定
# plt.xlabel('Iteration')
# plt.ylabel('Value')
# plt.title('s list')
# plt.legend()
# plt.show()


for i in range(num_usr):
    for j in range(num_objects[i]):
        y_values = [x_list[it][i][j] for it in range(iter)]
        plt.plot(range(iter), y_values, label=f'ID ({i}, {j})')

# グラフの設定
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('x list')
plt.legend()
plt.savefig(f"../graph/distributed_x_list_user_{num_usr}_step_{iter}.pdf")
# plt.show()

for i in range(num_usr):
    plt.plot(range(iter), scores[i])

# グラフの設定
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('score (distributed)') 
plt.legend()
plt.savefig(f"../graph/distributed_scores_user_{num_usr}_step_{iter}.pdf")
# plt.show()

# print(np.array(mu_list).shape)
plt.plot(range(iter), total_scores)

# グラフの設定
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('total score list (distributed)')
plt.legend()
plt.savefig(f"../graph/distributed_total_user_{num_usr}_step_{iter}.pdf")
# plt.show()

# max_capacities_flat, mc_indices, mc_lengths = flatten_and_index(max_capacities)
# for i in range(num_usr):
#         for j in range(i + 1, num_usr):
#             print(f"(User {i} and user {j}) max: {bandwidth[i][j]}, usage: {-g_ij(i, j, x[i], x[j])}")

for i in range(num_usr):
    print(f"User {i}'s parameter: ", x[i])

print_cap_constraint()

print(f"optimization execution time for {num_usr} users: ", optimization_time)

#実行時間をファイルに書き込む
with open(f'../result/time_distributed_{num_usr}users.txt', 'w') as file:
    for i in range(iter):
        file.write(f"{times_for_iter[i]} {total_scores[i]}\n")