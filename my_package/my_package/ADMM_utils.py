import numpy as np
import copy
import torch
import rclpy
from std_msgs.msg import Float32

def local_optimize(id, x, other_usages, mu, s, rho, optimizer, model, user_list, max_capacities, bandwidth, iter_inner = 3, ctx = torch.tensor(1)):   # ローカル最適化
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
        optimizer.zero_grad()  # 勾配のリセット
        
        param = x

        val = model(ctx.unsqueeze(0).float(), param.unsqueeze(0).float()) * (-1.0)
        
        # 他のユーザーとの制約を追加
        for j in user_list:
            other_usage = other_usages[j].clone().detach()
            # print(j)
            # print(mu[j])
            # print(other_usage)
            # print(f"bandwidth constraint between user {id} and {j}", bandwidth[j])
            # print(s[j])
            # print(val.dtype)
            # print(mu[j].dtype)
            # print((g_ij(id, param, max_capacities, other_usage, bandwidth[j]) + s[j]).dtype)
            gij_val = g_ij(id, param, max_capacities, other_usage, bandwidth[j])
            val = val + mu[j] * (gij_val + s[j])
            val = val + (0.5 * rho) * ((gij_val + s[j]) ** 2)

        loss = val.sum()
        
        # loss.backward(retain_graph=True)
        loss.backward()

        # print(f"user {id}'s parameter before update: {x}")
        optimizer.step()  # パラメータの更新
        # print(f"user {id}'s parameter after update: {x}")

        #パラメータのクリッピング
        with torch.no_grad():
            x.copy_(x.clamp(min=0.0, max=1.0))
            # x.clamp_(min=0.0, max=1.0)

        # if id == 7:
        #     print(f"User {id}'s parameter: ", x)

    return x.detach()


def g_ij(id, x, max_capacities, other_usage, bandwidth):
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
    dri = x[::2]
    fri = x[1::2]

    # num_objects[i] と max_capacities[i] に対応する部分のテンソル計算
    val_i = (max_capacities * dri * fri).sum() * scaling_factor
    # val_j = (max_capacities[scenes[j] - 1][:num_objects[j]] * drj * frj).sum() * scaling_factor

    # 合計値を計算してバンド幅を引く
    val = (val_i + other_usage - bandwidth)

    # print(f"user {id}'s ds rates: {dri}")
    # print(f"user {id}'s frame rates: {fri}")
    # print(f"other_usage: {other_usage}, bandwidth_constraint with the user: {bandwidth}")
    # print(f"user {id}'s constraint value: {val}")

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


def update_s(id, s, mu, x, other_usages, user_list, max_capacities, scene_i, rho, bandwidth):
    # id: ユーザID
    # s: 自身と周囲のユーザとのパラメータsの値
    # mu: μ、ラグランジュ関数の項の係数
    # x: 自身のパラメータxの値
    # other_usage: 近隣ユーザの通信路使用量
    # user_list: 近隣ユーザのリスト
    # max_capacities: 自身のオブジェクトの最大容量
    # scene_i: そのユーザのシーンID
    # rho: ρ、拡張ラグランジュ関数の項のウェイト

    # updated_s = s.clone().detach()
    updated_s = {user_id: tensor.detach().clone().requires_grad_(False) for user_id, tensor in s.items()}
    for i in user_list:
        # update_value = max(0, updated_s[i, j] - (g_ij(i, j, x[indexes[i] : indexes[i + 1]], x[indexes[j] : indexes[j + 1]]) + updated_s[i, j]) - (mu[i, j] / rho))
        update_value = updated_s[i] - (g_ij(id, x, max_capacities, other_usages[i], bandwidth[i]).detach() + updated_s[i]) - (mu[i] / rho)
        update_value = update_value.clamp(min=0)
        updated_s[i] = update_value
                
    return updated_s


def update_mu(id, s, x, mu, max_capacities, user_list, scene_i, other_usages, bandwidth):
    # id: ID
    # s: パラメータs
    # x: パラメータx
    # mu: パラメータμ
    # max_capacities: このユーザのオブジェクトの容量
    # scene_i: そのユーザのシーンID
    # other_usage: 他ユーザの通信路使用量

    # updated_mu = mu.clone().detach()
    updated_mu = {user_id: tensor.detach().clone().requires_grad_(False) for user_id, tensor in mu.items()}
    for i in user_list:
        # update_value = mu[i][j] + g_ij(i, j, x[indexes[i] : indexes[i + 1]], x[indexes[j] : indexes[j + 1]]) + s[i][j]
        update_value = mu[i] + g_ij(id, x, max_capacities, other_usages[i], bandwidth[i]).detach() + s[i]
        updated_mu[i] = update_value

    return updated_mu


def update_score(model, x, mu, s, x_list, mu_list, s_list, scores, file_path, time, ctx = torch.tensor(1)) :
    score = model(ctx.unsqueeze(0).float(), x.unsqueeze(0).float())
    score = score.detach().numpy()[0]
    scores.append(score)

    # x_list.append(copy.deepcopy( x.detach().numpy() ))
    x_list.append(copy.deepcopy(x.detach().numpy()))
    mu_list.append(copy.deepcopy({user_id: tensor.detach() for user_id, tensor in mu.items()}))
    s_list.append(copy.deepcopy({user_id: tensor.detach() for user_id, tensor in s.items()}))
    # total_scores.append(sum_score)

    # スコアをファイルに保存したい、後で追加
    with open(file_path, mode = 'a') as f:
        f.write(f"time: {time}, score: {score}\n")

    return scores, x_list, mu_list, s_list


def is_converged(scores, iter, window_size=25, tol=1e-7):   #収束判定
    ret1 = False
    if(iter > window_size):   # 過去一定ステップ（デフォルト：20）の移動平均が1e-8以下の場合収束
        ma_now = np.sum([scores[len(scores) - window_size : len(scores)]]) / window_size
        ma_previous = np.sum([scores[len(scores) - window_size - 1 : len(scores) - 1]]) / window_size
        # print("moving average: ", ma_now)
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


def publish_usage(node, parami, max_capacities, other_usage, pub):
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

    # トピックに配信するメッセージの作成
    msg = Float32()
    msg.data = float(val_i.item())

    # print(f"user {i}'s max capacity: {max_capacities}")
    # print(f"user {i}'s ds rate: {dri}")
    # print(f"user {i}'s frame rate: {fri}")
    # print(f"user {i}'s usage: {msg.data}")

    # トピックにパブリッシュ
    pub.publish(msg)
    # if node.user_id == 4:
    node.get_logger().info(f'Published from user_{node.user_id}: {msg.data}')