import re
import sys
import os

from RQ2_complexity_effect import get_cyclomatic_complexity
from RQ2_loc_effect import get_loc
from RQ2_vuln_type_effect import get_vuln_type

import os
import pandas as pd
import csv
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

results_root = "results"
data_root = "data"
datasets = [
    "smartbugs",
    "ReposVul_cpp",
    "ReposVul_py",
    "PrimeVul_c"
]

output_path = "RQs/RQ6_score_shift/score_shift_cases.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

rows = []


for dataset in tqdm(datasets, desc="Processing datasets"):
    dataset_eval_path = os.path.join(results_root, dataset, "evaluate")
    dataset_data_path = os.path.join(data_root, dataset, "original")
    

    if not os.path.isdir(dataset_eval_path):
        continue

    for model in tqdm(os.listdir(dataset_eval_path), desc=f"Processing models in {dataset}"):
        model_dir = os.path.join(dataset_eval_path, model)
        if not os.path.isdir(model_dir):
            continue

        # load original.csv
        ori_csv = os.path.join(model_dir, "original.csv")
        if not os.path.exists(ori_csv):
            continue

        df_ori = pd.read_csv(ori_csv)
        if "file_name" not in df_ori.columns or "eval_score" not in df_ori.columns:
            continue

        ori_score_map = df_ori.set_index("file_name")["eval_score"].to_dict()

        for file in os.listdir(model_dir):
            if file == "original.csv" or not file.endswith(".csv"):
                continue
            combo_name = file.replace(".csv","")
            df_obf = pd.read_csv(os.path.join(model_dir, file))
            for _, row in df_obf.iterrows():
                fname = row["file_name"]
                obf_score = row["eval_score"]
                ori_score = ori_score_map.get(fname)

                if pd.isna(obf_score) or pd.isna(ori_score):
                    continue

                try:
                    obf_score = int(obf_score)
                    ori_score = int(ori_score)
                except:
                    continue

                # 检测分数变动
                shift_type = None
                if ori_score in [3,4] and obf_score in [1,2]:
                    shift_type = "degrade"   # 攻击成功
                elif ori_score in [1,2] and obf_score in [3,4]:
                    shift_type = "upgrade"   # 混淆反而暴露漏洞
                else:
                    shift_type = 'other'  # 没有分数变动

                if shift_type is None:
                    continue

                code_path = os.path.join(dataset_data_path, fname)
                ori_loc = get_loc(code_path)
                ori_complexity = get_cyclomatic_complexity(code_path)
                obf_loc = get_loc(os.path.join(data_root, dataset, file.replace(".csv", ""), fname))
                obf_complexity = get_cyclomatic_complexity(os.path.join(data_root, dataset, file.replace(".csv", ""), fname))
                vuln_type = get_vuln_type(dataset, fname)

                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "combo_name": combo_name,
                    "file_name": fname,
                    "shift_type": shift_type,
                    "ori_score": ori_score,
                    "obf_score": obf_score,
                    "vuln_type": vuln_type,
                    "ori_loc": ori_loc,
                    "ori_complexity": ori_complexity,
                    "obf_loc": obf_loc,
                    "obf_complexity": obf_complexity
                })

if rows:
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"✅ 分数变动样本已保存：{output_path}")
else:
    print("⚠️ 没找到任何分数变动样本！")
