#ニューラルネットワークの性能評価を行う
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d

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
    

class myPICNN(nn.Module):
    def __init__(self, num_input, hidden_size_u, hidden_size_z, output_size = 1):
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
        # self.score_scaler = score_scaler

    def forward(self, x, y):

        # u1 = torch.relu(self.x_u1(x))
        u1 = self.activation(self.x_u1(x))
        # z1_1 = self.w0z( y * self.w0zu(x) )
        z1_2 = self.w0y( y * ( self.w0yu(x) ) )
        z1_3 = self.w0u(x)
        # z1 = torch.relu(z1_1 + z1_2 + z1_3)
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
            # 逆正規化を実行
            # original_output = self.score_scaler.inverse_transform(output.detach().numpy())
            # return torch.tensor(original_output)

        return output


#モデルの読み込み（シーン名は適宜変更）
model = myPICNN(22, 10, 100)
model.load_state_dict(torch.load('../model/model_scene_1.pth'))
model.eval()

# model = myPICNN(16, 10, 100)
# model.load_state_dict(torch.load('../model/model_scene_2.pth'))
# model.eval()

# model = myPICNN(30, 10, 100)
# model.load_state_dict(torch.load('../model/model_scene_3.pth'))
# model.eval()

# 正解のスコアと予測値を保存するリスト
true_scores = []
predicted_scores = []

csv_file = '../dataset/dataset_test_scene1.csv'   #シーン名は適宜変更
# csv_file = '../dataset/dataset_test_scene2.csv'
# csv_file = '../dataset/dataset_test_scene3.csv'
dataset2 = CustomDataset2(csv_file)
dataloader2 = DataLoader(dataset2, batch_size=64, shuffle=True)

with torch.no_grad():
    for inputs, targets in dataloader2:  # dataloaderは評価用データ
        outputs = model(inputs[:, :1], inputs[:, 1:])  # モデルの予測
        # targets = dataset2.score_scaler.inverse_transform(targets.detach().cpu().numpy())  #本来のスコアに戻す
        true_scores.extend(targets.numpy().flatten())
        # true_scores.extend(targets.numpy())
        predicted_scores.extend(outputs.numpy().flatten())

true_scores = np.array(true_scores)
predicted_scores = np.array(predicted_scores)
count = 0   #誤差5%いないのデータの数

residual = np.array([y_test - y_pred for y_test, y_pred in zip(true_scores, predicted_scores)])   #残差

SE = np.sqrt(np.sum(residual**2) / (len(true_scores) - 2))

confidence_level = 0.95
degrees_of_freedom = len(true_scores) - 2
t_value = stats.t.ppf((1 + confidence_level) / 2., degrees_of_freedom)
margin_of_error = t_value * SE

# 信頼区間の上限と下限
ci_upper = predicted_scores + margin_of_error
ci_lower = predicted_scores - margin_of_error

# 95%信頼区間内のデータ数をカウント
within_ci = np.sum((true_scores >= ci_lower) & (true_scores <= ci_upper))

print(within_ci)

for i in range(2000):
    if (predicted_scores[i] >= 0.90 * true_scores[i]) and (predicted_scores[i] <= 1.10 * true_scores[i]):
        count += 1

print("Number of data error is less than 10%: ", count)
print("Rate of data error is less than 10%: ", count / 2000)

# フォントの種類を指定
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16

# 散布図をプロット
plt.figure(figsize=(8, 8))
plt.plot([min(true_scores), max(true_scores)], [min(true_scores), max(true_scores)], color='red', linestyle='--', linewidth=2)  # 参考線
plt.plot([min(true_scores), max(true_scores)], [min(true_scores) + margin_of_error, max(true_scores) + margin_of_error], color='green', linestyle='--', linewidth=2)  # 参考線
plt.plot([min(true_scores), max(true_scores)], [min(true_scores) - margin_of_error, max(true_scores) - margin_of_error], color='green', linestyle='--', linewidth=2)  # 参考線
plt.scatter(true_scores, predicted_scores, alpha=0.4, s=16)
plt.xlabel('True Scores')
plt.ylabel('Predicted Scores')
plt.savefig("../graph/scene1_model_eval.png")   #シーン名は適宜変更
plt.show()