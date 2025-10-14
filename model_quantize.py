import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import os
from collections import OrderedDict

# 元のPICNN.pyはインポート不要です

# --- Step 1: 量子化専用の、完全に独立したモデルクラスを定義 ---
# このクラスは標準のnn.Linearのみで構成され、forwardパスもクリーンです。
class QuantizableMyPICNN(nn.Module):
    def __init__(self, num_input, hidden_size_u, hidden_size_z):
        super(QuantizableMyPICNN, self).__init__()
        
        # すべてのレイヤーを標準のnn.Linearで定義
        self.x_u1 = nn.Linear(1, hidden_size_u)
        self.w0zu = nn.Linear(1, num_input)
        self.w0yu = nn.Linear(1, num_input)
        self.w0z = nn.Linear(num_input, hidden_size_z)
        self.w0y = nn.Linear(num_input, hidden_size_z)
        self.w0u = nn.Linear(1, hidden_size_z)

        self.u1_u2 = nn.Linear(hidden_size_u, hidden_size_u)
        self.w1zu = nn.Linear(hidden_size_u, hidden_size_z)
        self.w1yu = nn.Linear(hidden_size_u, num_input)
        self.w1z = nn.Linear(hidden_size_z, hidden_size_z)
        self.w1y = nn.Linear(num_input, hidden_size_z)
        self.w1u = nn.Linear(hidden_size_u, hidden_size_z)

        self.u2_u3 = nn.Linear(hidden_size_u, hidden_size_u)
        self.w2zu = nn.Linear(hidden_size_u, hidden_size_z)
        self.w2yu = nn.Linear(hidden_size_u, num_input)
        self.w2z = nn.Linear(hidden_size_z, hidden_size_z)
        self.w2y = nn.Linear(num_input, hidden_size_z)
        self.w2u = nn.Linear(hidden_size_u, hidden_size_z)
        
        self.output = nn.Linear(hidden_size_z, 1)
        self.activation = nn.ReLU()

    def forward(self, x, y):
        # clampや重みへのアクセスを一切含まない、標準的なforwardパス
        u1 = self.activation(self.x_u1(x))
        z1_2 = self.w0y( y * ( self.w0yu(x) ) )
        z1_3 = self.w0u(x)
        z1 = self.activation(z1_2 + z1_3)

        u2 = self.activation(self.u1_u2(u1))
        z2_1 = self.w1z( z1 * self.activation(self.w1zu(u1)) )
        z2_2 = self.w1y( y * ( self.w1yu(u1)) )
        z2_3 = self.w1u(u1)
        z2 = self.activation(z2_1 + z2_2 + z2_3)

        u3 = self.activation(self.u2_u3(u2))
        z3_1 = self.w2z( z2 * self.activation(self.w2zu(u2)) )
        z3_2 = self.w2y( y * ( self.w2yu(u2)) )
        z3_3 = self.w2u(u2)
        z3 = self.activation(z3_1 + z3_2 + z3_3)

        output = self.output(z2)
        return output

def quantize_model(model_path, num_input, hidden_size_u, hidden_size_z):
    
    # --- Step 2: 量子化専用モデルのインスタンスを作成 ---
    model_fp32 = QuantizableMyPICNN(
        num_input=num_input, 
        hidden_size_u=hidden_size_u, 
        hidden_size_z=hidden_size_z
    )
    
    # --- Step 3: state_dictのロード、キー名変更、重みクランプをすべて実行 ---
    state_dict_original = torch.load(model_path, map_location=torch.device('cpu'))
    
    # キーの名前を変更（.linear. を . に置換）
    state_dict_renamed = OrderedDict()
    for key, value in state_dict_original.items():
        if ".linear." in key:
            new_key = key.replace(".linear.", ".")
            state_dict_renamed[new_key] = value
        else:
            state_dict_renamed[key] = value

    # 非負制約を適用
    keys_to_clamp = ["w0z.weight", "w1z.weight", "w2z.weight", "output.weight"]
    for key in keys_to_clamp:
        if key in state_dict_renamed:
            state_dict_renamed[key] = state_dict_renamed[key].clamp(min=0)
            print(f"重み '{key}' をクランプしました。")

    # 修正済みのstate_dictをモデルにロード
    model_fp32.load_state_dict(state_dict_renamed)
    print(f"学習済みモデルを正常にロードし、重みを修正しました: {model_path}")

    # --- Step 4: 量子化プロセスを実行（変更なし） ---
    model_fp32.eval()
    backend = "onednn"
    model_fp32.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_prepared = torch.quantization.prepare(model_fp32)

    print("キャリブレーションを開始します...")
    calibration_data_count = 100
    with torch.no_grad():
        dummy_x = torch.randn(calibration_data_count, 1)
        dummy_y = torch.randn(calibration_data_count, num_input)
        for i in range(calibration_data_count):
            model_prepared(dummy_x[i].unsqueeze(0), dummy_y[i].unsqueeze(0))
    print("キャリブレーションが完了しました。")

    model_int8 = torch.quantization.convert(model_prepared)
    print("モデルをINT8に変換しました。")
    
    # 保存とサイズ比較
    base, ext = os.path.splitext(model_path)
    quantized_model_path = f"{base}_quantized.pth"
    scripted_quantized_model = torch.jit.script(model_int8)
    torch.jit.save(scripted_quantized_model, quantized_model_path)
    print(f"量子化モデルを保存しました: {quantized_model_path}")
    
    original_size = os.path.getsize(model_path)
    quantized_size = os.path.getsize(quantized_model_path)
    print("\n--- サイズ比較 ---")
    print(f"オリジナルモデルのサイズ: {original_size / 1024:.2f} KB")
    print(f"量子化モデルのサイズ:   {quantized_size / 1024:.2f} KB")
    print(f"サイズ削減率: {100 * (1 - quantized_size / original_size):.2f}%")


if __name__ == '__main__':
    # モデルのパスとパラメータを設定
    MODEL_PATH = 'ADMM_with_ICNN_on_ROS/model/model_scene_3.pth'
    NUM_INPUT = 30
    HIDDEN_SIZE_U = 10
    HIDDEN_SIZE_Z = 100
    
    # 既存のPICNN.pyは不要なので、テスト用にダミーファイルを作成する処理は削除してOKです
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイルが見つかりません: {MODEL_PATH}")
        exit() # 存在しない場合は終了
        
    quantize_model(MODEL_PATH, NUM_INPUT, HIDDEN_SIZE_U, HIDDEN_SIZE_Z)