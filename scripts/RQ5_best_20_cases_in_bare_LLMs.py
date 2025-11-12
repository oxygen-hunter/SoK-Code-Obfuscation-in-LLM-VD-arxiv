
'''
从"RQs/RQ6_score_shift/count_up_down_on_single_obf_x_all_models/upgrade_count.csv"中选出在所有obf上产生upgrade数量之和最多的20个样本，保证每个数据集5个
'''
'''
参考格式如下
dataset,file_name,combo_C1,combo_C2,combo_C3,combo_D1+D2+D3,combo_D4+D5+D6+D7,combo_D8,combo_L1,combo_L2,combo_L3,combo_L4+L5+L6,combo_L7,combo_L8
smartbugs,18_access_control.sol,5,4,2,2,2,4,3,2,2,1,1,5
smartbugs,19_access_control.sol,2,2,1,0,1,1,1,1,2,2,1,2
'''

import os
import pandas as pd
import shutil
from pathlib import Path

# 文件路径
csv_path = "RQs/RQ6_score_shift/count_up_down_on_single_obf_x_all_models/upgrade_count.csv"
data_dir = "data"

# 读取 CSV 文件
df = pd.read_csv(csv_path)

# 计算每行的 upgrade 数量之和（从 combo_C1 到 combo_L8）
df['total_upgrade'] = df.iloc[:, 2:].sum(axis=1)

# 按数据集分组，选出每个数据集中 upgrade 数量最多的 5 个样本
top_per_dataset = df.groupby('dataset', group_keys=False).apply(lambda x: x.nlargest(5, 'total_upgrade'))

# 从所有数据集中选出总和最多的 20 个样本
top_20_samples = top_per_dataset.nlargest(20, 'total_upgrade')

# 输出选出的样本
print("Top 20 samples:")
print(top_20_samples)

# 遍历选出的样本，复制文件到目标目录
for _, row in top_20_samples.iterrows():
    dataset = row['dataset']
    file_name = row['file_name']
    
    # 定义源目录和目标目录
    source_dir = Path(data_dir) / dataset
    target_dir = Path(data_dir) / f"{dataset}_upgrade_top20"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    
    # 复制出现在csv中的 original 和 obfuscation 文件
    obf_combos = [d for d in source_dir.iterdir() if d.is_dir() and (d.name in df.columns[2:] or d.name == "original")]
    for obf_combo in obf_combos:
        obf_file = obf_combo / file_name
        if obf_file.exists():
            combo_target_dir = target_dir / obf_combo.name
            combo_target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(obf_file, combo_target_dir)
    
    # 复制 explanation 文件
    explanation_file = source_dir / "explanation" / f"{file_name}.txt"
    if explanation_file.exists():
        explanation_target_dir = target_dir / "explanation"
        explanation_target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(explanation_file, explanation_target_dir)

# 保存选出的样本到 CSV 文件
output_csv_path = "RQs/RQ6_score_shift/count_up_down_on_single_obf_x_all_models/top_20_upgrade_samples.csv"
top_20_samples.to_csv(output_csv_path, index=False)
print(f"Top 20 samples saved to {output_csv_path}")