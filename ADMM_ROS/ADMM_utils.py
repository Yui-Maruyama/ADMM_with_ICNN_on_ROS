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

def local_optimize(id, x, other_usages, mu, s, rho, optimizer, model, user_list, max_capacities, iter_inner = 1, ctx = torch.tensor(1)):   # ローカル最適化
    # id: そのユーザのID
    # x: そのユーザのオブジェクトのパラメータ
    # other_usages: 他ユーザ（近隣ユーザ全員分）の
    # mu: そのユーザと近隣ユーザ間のμ
    # s: そのユーザと近隣ユーザ間のs
    # optimizer: オプティマイザ（ユーザごとに定義されている）
    # model: そのユーザのモデル（ニューラル関数）
    # user_list: 近隣ユーザのリスト（この番号をインデックスとして色々なデータにアクセス）
    # max_capacities: 自身の空間のオブジェクトの最大容量

    for i in range(iter_inner):
        # optimization_start_time = time.time()

        val = 0
        optimizer.zero_grad()  # 勾配のリセット
        
        param = x

        val = model(ctx.unsqueeze(0).float(), param.unsqueeze(0).float()) * (-1.0)
        
        # 他のユーザーとの制約を追加
        for j in user_list:
            other_usage = other_usages[j].clone().detach()
            val = val + mu[j] * (g_ij(id, param, max_capacities[i], other_usages[j]) + s[id][j])
            val = val + (0.5 * rho) * ((g_ij(id, param, max_capacities[i], other_usages[j]) + s[id][j]) ** 2)

        loss = val.sum()
        
        loss.backward()

        optimizer.step()  # パラメータの更新

        #パラメータのクリッピング
        with torch.no_grad():
            x.copy_(param)
            x.clamp_(min=0.0, max=1.0)

        return x


def g_ij(i, parami, max_capacities, other_usage, bandwidth):
    # i:ユーザID
    # parami: そのユーザのパラメータ
    # max_capacities: そのユーザの空間のオブジェクトの最大容量のリスト
    # scene_i: そのユーザのシーンID（シーンによりオブジェクトの種類、個数が変わる）
    # other_usage: 他ユーザ（ここではユーザjのみ）の通信路使用量合計（KB/s）
    # bandwidth: 対象ユーザとの間の通信路の最大容量

    # 定数の計算を一度だけ行う（単位を揃えるため？要検証）
    scaling_factor = 10.0 / 1000

    # dri, fri, drj, frj をテンソルとして取得
    # drはダウンサンプリングレート、frはフレームレート（元のパラメータでは交互に置かれている）
    dri = parami[::2]
    fri = parami[1::2]

    # num_objects[i] と max_capacities[i] に対応する部分のテンソル計算
    val_i = (max_capacities * dri * fri).sum() * scaling_factor
    # val_j = (max_capacities[scenes[j] - 1][:num_objects[j]] * drj * frj).sum() * scaling_factor

    # 合計値を計算してバンド幅を引く
    val = (val_i + other_usage - bandwidth)

    return val


# def update_x(x):
#     global optimization_time
#     # newx = []
#     # x_old = x.clone().detach()
#     x_old = copy.deepcopy(x)
#     x_new = []
#     optimization_times = []
#     for id in range(num_usr):
#         optimization_start_time = time.time()

#         minimize(id)

#         #ユーザの最適化が全部終わるまでパラメータ更新しない
#         x_new.append(x[id].clone().detach())
#         x[id].data = x_old[id].clone().detach()

#         optimization_end_time = time.time()
#         optimization_times.append(optimization_end_time - optimization_start_time)

#     for id in range(num_usr):
#         # x.data[indexes[id] : indexes[id + 1]] = x_new[id][indexes[id] : indexes[id + 1]]
#         x[id].data = x_new[id]

#     #このイテレーションにかかった時間（ユーザごとの実行時間のうち最大のもの）
#     times_for_iter.append(max(optimization_times))
#     optimization_time += sum(optimization_times)


def update_s(id, s, mu, x, other_usages, user_list, max_capacities, scene_i, rho):
    # id: ユーザID
    # s: 自身と周囲のユーザとのパラメータsの値
    # mu: μ、ラグランジュ関数の項の係数
    # x: 自身のパラメータxの値
    # other_usage: 近隣ユーザの通信路使用量
    # user_list: 近隣ユーザのリスト
    # max_capacities: 自身のオブジェクトの最大容量
    # scene_i: そのユーザのシーンID
    # rho: ρ、拡張ラグランジュ関数の項のウェイト

    updated_s = s.clone().detach()
    for i in user_list:
        # update_value = max(0, updated_s[i, j] - (g_ij(i, j, x[indexes[i] : indexes[i + 1]], x[indexes[j] : indexes[j + 1]]) + updated_s[i, j]) - (mu[i, j] / rho))
        update_value = max(0, updated_s[i] - (g_ij(id, x, max_capacities, scene_i, other_usages[i]) + updated_s[i]) - (mu[i] / rho))
        updated_s[i] = update_value
    
    # print(g_ij(0, 1, x[0].clone().detach(), x[1].clone().detach()), updated_s)
    # print(updated_s)
                
    return updated_s


def update_mu(id, s, x, mu, max_capacities, user_list, scene_i, other_usages):
    # id: ID
    # s: パラメータs
    # x: パラメータx
    # mu: パラメータμ
    # max_capacities: このユーザのオブジェクトの容量
    # scene_i: そのユーザのシーンID
    # other_usage: 他ユーザの通信路使用量

    updated_mu = mu.clone().detach()
    for i in user_list:
        # update_value = mu[i][j] + g_ij(i, j, x[indexes[i] : indexes[i + 1]], x[indexes[j] : indexes[j + 1]]) + s[i][j]
        update_value = mu[i] + g_ij(id, x, max_capacities, scene_i, other_usages[i]) + s[i]
        updated_mu[i] = update_value

        # print(g_ij(i, j, x[i].clone().detach(), x[j].clone().detach()) + s[i][j])

    return updated_mu


def update_score(model, x, mu, s, x_list, mu_list, s_list, scores, file_path, time, ctx = torch.tensor(1)) :
    score = model(ctx.unsqueeze(0).float(), x.unsqueeze(0).float())
    score = score.detach().numpy()[0]
    scores.append(score)

    # x_list.append(copy.deepcopy( x.detach().numpy() ))
    x_list.append(copy.deepcopy( x.detach().numpy() ))
    mu_list.append(copy.deepcopy(mu.detach().numpy()))
    s_list.append(copy.deepcopy(s.detach().numpy()))
    # total_scores.append(sum_score)

    # スコアをファイルに保存したい、後で追加
    with open(file_path, mode = 'a') as f:
        f.write(f"{time}, {score}")

    return scores, x_list, mu_list, s_list


def is_converged(scores, iter, window_size=20, tol=1e-8):   #収束判定
    ret1 = False
    if(iter > window_size):   # 過去一定ステップ（デフォルト：20）の移動平均が1e-8以下の場合収束
        ma_now = np.sum([scores[len(scores) - window_size : len(scores)]]) / window_size
        ma_previous = np.sum([scores[len(scores) - window_size - 1 : len(scores) - 1]]) / window_size
        print("moving average: ", ma_now)
        if (abs(ma_now / ma_previous - 1) < tol and ma_now > 0):
            ret1 = True
    else:
        return False
    
    # 最大値を20ステップ間更新できないと収束
    max_score = max(scores)
    max_index = scores.index(max_score)
    max_not_updated_count = iter - max_index

    ret2 = max_not_updated_count > window_size

    return ret1 and ret2


def publish_usage(i, parami, max_capacities, other_usage, pub):
    # i:ユーザID
    # parami: そのユーザのパラメータ
    # max_capacities: そのユーザの空間のオブジェクトの最大容量のリスト
    # scene_i: そのユーザのシーンID（シーンによりオブジェクトの種類、個数が変わる）
    # other_usage: 他ユーザ（ここではユーザjのみ）の通信路使用量合計（KB/s）
    # pub: パブリッシャ、データを配信する場所

    # dri, fri, drj, frj をテンソルとして取得
    # drはダウンサンプリングレート、frはフレームレート（元のパラメータでは交互に置かれている）
    dri = parami[::2]
    fri = parami[1::2]

    # num_objects[i] と max_capacities[i] に対応する部分のテンソル計算
    val_i = (max_capacities * dri * fri).sum()
    # val_j = (max_capacities[scenes[j] - 1][:num_objects[j]] * drj * frj).sum() * scaling_factor

    # 合計値を計算
    val = val_i + other_usage

    # トピックにパブリッシュ
    pub.publish(val)