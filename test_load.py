import torch

# 量子化済みモデルのパス
# 実際のパスに合わせて修正してください
quantized_model_path = './model/model_scene_1_quantized.pth'

print(f"Loading quantized model from: {quantized_model_path}")

# torch.jit.loadでモデルをロード
model = torch.jit.load(quantized_model_path)
model.eval()

print("Model loaded successfully.")

# ダミーデータで推論を実行
# モデルの入力サイズに合わせて修正してください
dummy_x = torch.randn(1, 1)
dummy_y = torch.randn(1, 22) # NUM_INPUT

print("Running inference with dummy data...")
with torch.no_grad():
    output = model(dummy_x, dummy_y)

print("Inference successful!")
print("Output:", output)