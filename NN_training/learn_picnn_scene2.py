import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

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
    

# 1. カスタムデータセットクラスの作成
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=0)  # CSVファイルからデータを読み込み

        # スコアの正規化をここで実施
        # self.score_scaler = StandardScaler()
        # self.data['score'] = self.score_scaler.fit_transform(self.data[['score']])

        # スコア以外のすべての列を抽出して正規化
        for column in self.data.columns:
            if "downsampling rate" in column:
                self.data[column] = self.data[column] / 100
            elif "frame rate" in column:
                self.data[column] = self.data[column] / 10

    def __len__(self):
        return len(self.data)  # データセットのサンプル数を返す

    def __getitem__(self, idx):
        # パラメータとスコアの抽出
        parameters = self.data.iloc[idx, :-1].values  # 最後の列を除くすべての列をパラメータとして取得
        score = self.data.iloc[idx, -1]  # 最後の列をスコアとして取得
        
        # Tensorに変換
        parameters = torch.tensor(parameters, dtype=torch.float32)
        score = torch.tensor(score, dtype=torch.float32)
        score = score.unsqueeze(0)  # スコアの形状を [64, 1] に変更（NNの出力は[64, 1]なので）
        
        return parameters, score
    

class CustomDataset2(Dataset):   #テストデータ用
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=0)  # CSVファイルからデータを読み込み

        # スコアの正規化をここで実施
        # self.score_scaler = StandardScaler()
        # self.data['score'] = self.score_scaler.fit_transform(self.data[['score']])

        # スコア以外のすべての列を抽出して正規化
        for column in self.data.columns:
            if "downsampling rate" in column:
                self.data[column] = self.data[column] / 100
            elif "frame rate" in column:
                self.data[column] = self.data[column] / 10

    def __len__(self):
        return len(self.data)  # データセットのサンプル数を返す

    def __getitem__(self, idx):
        # パラメータとスコアの抽出
        parameters = self.data.iloc[idx, :-1].values  # 最後の列を除くすべての列をパラメータとして取得
        score = self.data.iloc[idx, -1]  # 最後の列をスコアとして取得
        
        # Tensorに変換
        parameters = torch.tensor(parameters, dtype=torch.float32)
        score = torch.tensor(score, dtype=torch.float32)
        score = score.unsqueeze(0)  # スコアの形状を [64, 1] に変更（NNの出力は[64, 1]なので）
        
        return parameters, score
    

class PICNN_scene1(nn.Module):
    def __init__(self, num_input, hidden_size_u, hidden_size_z, score_scaler = None, output_size = 1):
        super(PICNN_scene1, self).__init__()

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
        # self.score_scaler = score_scaler

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

        # if self.score_scaler is not None and not self.training:
        #     # 逆正規化を実行
        #     original_output = self.score_scaler.inverse_transform(output.detach().numpy())
        #     return torch.tensor(original_output)

        return output
    
    def check_for_nan(self, tensor, layer_name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {layer_name}")



def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# モデル、損失関数、最適化手法の定義
num_obj = 8
num_input = 2 * num_obj   #動かないオブジェクトのフレームレートあり
# num_input = 2 + num_obj   #なし
hidden_size_u = 10
hidden_size_z = 100
output_size = 1

# データセットのファイルパスを指定
csv_file = '../dataset/dataset_train_scene2.csv'   #データ数10000
# csv_file = './videos/7090/dataset_no_framerate.csv'   #データ数10000
dataset = CustomDataset(csv_file)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

model = PICNN_scene1(num_input, hidden_size_u, hidden_size_z)
# model = PICNN_scene1(num_input, hidden_size_u, hidden_size_z)
criterion = nn.HuberLoss(delta=1.0)   #回帰問題にはこれ
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.apply(init_weights)

# 損失値を記録するリスト
train_losses = []

# 学習ループ
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for inputs, score in dataloader:
        optimizer.zero_grad()

        # print(inputs[0], score[0])

        # 順伝播
        outputs = model(inputs[:, :1], inputs[:, 1:])
        loss = criterion(outputs, score)

        # 逆伝播と最適化
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        epoch_loss += loss.item() * inputs.size(0)

        optimizer.step()

    # エポックごとの平均損失
    avg_loss = epoch_loss / 10000
    train_losses.append(avg_loss)
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}')

print('学習完了')

# 損失値の可視化
# plt.ylim(0, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.legend()
plt.savefig("../graph/learning_loss_scene2.png")
plt.show()

# モデルを評価モードに
model.eval()

# 正解のスコアと予測値を保存するリスト
true_scores = []
predicted_scores = []

csv_file = '../dataset/dataset_test_scene2.csv'   #データ数2000
dataset2 = CustomDataset2(csv_file)
dataloader2 = DataLoader(dataset2, batch_size=64, shuffle=True)

with torch.no_grad():
    for inputs, targets in dataloader2:  # dataloaderは評価用データ
        outputs = model(inputs[:, :1], inputs[:, 1:])  # モデルの予測
        # targets = dataset2.score_scaler.inverse_transform(targets.detach().cpu().numpy())  #本来のスコアに戻す
        true_scores.extend(targets.numpy())
        # true_scores.extend(targets.numpy())
        predicted_scores.extend(outputs.numpy())

true_scores = np.array(true_scores)
predicted_scores = np.array(predicted_scores)

# 散布図をプロット
plt.figure(figsize=(8, 8))
plt.scatter(true_scores, predicted_scores, alpha=0.5)
plt.plot([min(true_scores), max(true_scores)], [min(true_scores), max(true_scores)], color='red', linestyle='--')  # 参考線
plt.plot([min(true_scores), max(true_scores)], [min(true_scores) * 1.05, max(true_scores) * 1.05], color='green', linestyle='--')  # 参考線
plt.plot([min(true_scores), max(true_scores)], [min(true_scores) * 0.95, max(true_scores) * 0.95], color='green', linestyle='--')  # 参考線
plt.xlabel('True Scores')
plt.ylabel('Predicted Scores')
plt.title('True vs Predicted Scores')
plt.savefig("../graph/learning_result_scene2.png")
plt.show()

# 学習後、モデルの重みを保存する
torch.save(model.state_dict(), '../model/model_scene_2.pth')