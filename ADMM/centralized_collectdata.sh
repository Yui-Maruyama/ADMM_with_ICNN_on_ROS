# 初期値
start=5

# 終了値
end=30

# Pythonスクリプト名
script="./centralized_time_score_args.py"

# ループで引数を増やしながら実行
for (( i=$start; i<=$end; i++ ))
do
  # Pythonスクリプトを実行し、引数としてiを渡す
  python3 $script $i
done