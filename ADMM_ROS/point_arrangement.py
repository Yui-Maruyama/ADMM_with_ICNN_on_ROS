import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# パラメータ
num_points = 10
min_dist = 1.0
max_dist = 50.0
neighbor_threshold = 20.0  # 距離10以下の点を記録
max_attempts_per_point = 1000
area_size = 100

random.seed(114514)

# 点リスト初期化 (1点目を固定)
points = [np.array([0.0, 0.0])]

# 残りの点を逐次配置
for i in range(1, num_points):
    success = False
    for attempt in range(max_attempts_per_point):
        candidate = np.array([random.uniform(0, area_size), random.uniform(0, area_size)])
        distances = cdist([candidate], points)[0]
        if np.all((distances >= min_dist) & (distances <= max_dist)):
            points.append(candidate)
            success = True
            break
    if not success:
        raise Exception(f"Point {i} を配置できませんでした。")

print(f"{len(points)}個の点を配置できました！")

# 可視化
points = np.array(points)
plt.figure(figsize=(8, 8))
plt.scatter(points[:, 0], points[:, 1], c='blue', s=30)

for i in range(len(points)):
    for j in range(i + 1, len(points)):
        dist = np.linalg.norm(points[i] - points[j])
        if min_dist <= dist <= max_dist:
            plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'gray', alpha=0.1)

plt.title('Point Placement with Distance Constraints')
plt.xlim(-area_size, area_size)
plt.ylim(-area_size, area_size)
plt.gca().set_aspect('equal')
plt.show()

# ✅ 改善2: 近い点のリストを出力
distances = cdist(points, points)
np.fill_diagonal(distances, np.inf)  # 自分自身との距離は除外

with open(f"../neighbors_{num_points}.txt", "w") as f:
    for i in range(len(points)):
        neighbors = [str(j) for j in range(len(points)) if distances[i][j] <= neighbor_threshold]
        line = f", ".join(neighbors) + "\n"
        f.write(line)

print("近接点リストを neighbors.txt に出力しました。")