#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据小数据集 (e.g. *_downgrade_top20) 中的 GitHubCopilot 或 Codex 的 combo.csv，
从大数据集的其他模型结果里筛选相同 file_name 的记录写出。
"""

import os
import pandas as pd
import sys
import os

# 添加项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.constants import LOCAL_AUDITORS, API_AUDITORS, AGENT_AUDITORS

BASE_DIR = "results"
DATASETS = ["smartbugs", "ReposVul_cpp", "ReposVul_py", "PrimeVul_c"]


grades = ["downgrade_top20", "upgrade_top20"]
for grade in grades:
    for dataset in DATASETS:
        print(f"\n=== Processing dataset: {dataset} ===")
        src_root = os.path.join(BASE_DIR, dataset, "evaluate")
        dst_root = os.path.join(BASE_DIR, f"{dataset}_{grade}", "evaluate")

        if not os.path.exists(src_root) or not os.path.exists(dst_root):
            print(f"⚠️ Missing source or target dir: {src_root} / {dst_root}")
            continue

        # 找到参考的5条样本（优先取 GitHubCopilot，否则 Codex）
        ref_df = None
        for model in AGENT_AUDITORS:
            ref_model_dir = os.path.join(dst_root, model)
            if not os.path.exists(ref_model_dir):
                continue
            for combo_file in os.listdir(ref_model_dir):
                if combo_file.endswith(".csv"):
                    ref_path = os.path.join(ref_model_dir, combo_file)
                    ref_df = pd.read_csv(ref_path)
                    if "file_name" in ref_df.columns:
                        ref_file_names = set(ref_df["file_name"].astype(str))
                        print(f"🧩 Found reference combo {combo_file} with {len(ref_file_names)} files from {model}")
                        break
            if ref_df is not None:
                break

        if ref_df is None:
            print(f"❌ No reference found for {dataset}_{grade}")
            continue

        # 获取源模型目录（所有非 agent auditors）
        models = [m for m in os.listdir(src_root)
                if os.path.isdir(os.path.join(src_root, m)) and m not in AGENT_AUDITORS]

        for model in models:
            src_model_dir = os.path.join(src_root, model)
            dst_model_dir = os.path.join(dst_root, model)
            os.makedirs(dst_model_dir, exist_ok=True)

            for combo_file in os.listdir(src_model_dir):
                if not combo_file.endswith(".csv"):
                    continue
                src_csv = os.path.join(src_model_dir, combo_file)
                dst_csv = os.path.join(dst_model_dir, combo_file)

                try:
                    df = pd.read_csv(src_csv)
                except Exception as e:
                    print(f"⚠️ Failed to read {src_csv}: {e}")
                    continue

                if "file_name" not in df.columns:
                    print(f"⚠️ Skip {src_csv} (no 'file_name' column)")
                    continue

                subset_df = df[df["file_name"].astype(str).isin(ref_file_names)]
                subset_df.to_csv(dst_csv, index=False)
                print(f"✅ {model}/{combo_file}: wrote {len(subset_df)} rows")

print("\n🎉 完成所有小数据集的 LOCAL/API 模型结果过滤复制。")
