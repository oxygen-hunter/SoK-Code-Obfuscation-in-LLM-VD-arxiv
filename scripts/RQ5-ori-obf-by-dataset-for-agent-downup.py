import sys
import os
import pandas as pd
import numpy as np
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.constants import LLM_ABBR_MAPPING, SERIES_GROUPS

# 参数设置
mode = "upgrade"  # 或 "upgrade"
grades = ["downgrade_top20", "upgrade_top20"]
grade = grades[0] if mode == "downgrade" else grades[1]
results_root = "results"
datasets = [f"smartbugs_{grade}", f"ReposVul_cpp_{grade}", f"ReposVul_py_{grade}", f"PrimeVul_c_{grade}"]

SERIES_ORDER = ["qwen", "llama", "deepseek", "openai", "agent"]

datasets_macro = {
    datasets[0]: "\\smartbugs",
    datasets[1]: "\\primevulc",
    datasets[2]: "\\reposvulcpp",
    datasets[3]: "\\reposvulpy"
}

def get_sorted_models():
    ordered = []
    for series in SERIES_ORDER:
        ordered.extend(SERIES_GROUPS[series])
    return ordered

def shorten_model_name(model):
    return LLM_ABBR_MAPPING.get(model, model)

def sort_combo_key(name):
    if name == "original":
        return (0, 0)
    if "combo_L" in name:
        return (1, int(re.search(r"L(\d+)", name).group(1)))
    if "combo_D" in name:
        return (2, int(re.search(r"D(\d+)", name).group(1)))
    if "combo_C" in name:
        return (3, int(re.search(r"C(\d+)", name).group(1)))
    return (4, 0)


def compute(dataset, eval_mode):
    dataset_eval_path = os.path.join(results_root, dataset, "evaluate")
    if not os.path.isdir(dataset_eval_path):
        print(f"⚠️ 跳过：{dataset_eval_path} 不存在")
        return

    combo_stats = {}

    for model in os.listdir(dataset_eval_path):
        model_dir = os.path.join(dataset_eval_path, model)
        if not os.path.isdir(model_dir):
            continue

        # 找 original
        original_path = os.path.join(model_dir, "original.csv")
        if not os.path.exists(original_path):
            continue

        try:
            df_orig = pd.read_csv(original_path)
        except Exception as e:
            print(f"❌ 无法读取 {original_path}: {e}")
            continue

        # 建立原始分数字典
        orig_scores = dict(zip(df_orig["file_name"], df_orig["eval_score"]))

        for csv_file in os.listdir(model_dir):
            if not csv_file.endswith(".csv") or csv_file == "original.csv":
                continue

            combo = csv_file.replace(".csv", "")
            csv_path = os.path.join(model_dir, csv_file)

            try:
                df_combo = pd.read_csv(csv_path)
            except Exception as e:
                print(f"❌ 无法读取 {csv_path}: {e}")
                continue

            if eval_mode == "exist":
                succ_range = [2, 3, 4]
                fail_range = [1]
            elif eval_mode == "type":
                succ_range = [3, 4]
                fail_range = [1, 2]

            downgrade = upgrade = total = 0

            for _, row in df_combo.iterrows():
                fname = row["file_name"]
                if fname not in orig_scores:
                    continue

                try:
                    orig_score = int(orig_scores[fname])
                    combo_score = int(row["eval_score"])
                except:
                    continue

                if mode == "downgrade":
                    if orig_score in succ_range and combo_score in fail_range:
                        downgrade += 1
                elif mode == "upgrade":
                    if orig_score in fail_range and combo_score in succ_range:
                        upgrade += 1
                total += 1

            if combo not in combo_stats:
                combo_stats[combo] = {}
            if mode == "downgrade":
                combo_stats[combo][model] = downgrade / total if total > 0 else 0
            else:
                combo_stats[combo][model] = upgrade / total if total > 0 else 0

    if not combo_stats:
        print(f"⚠️ 数据集 {dataset} 没有结果。")
        return

    # 构造输出 DataFrame
    models_ordered = get_sorted_models()
    models_ordered_abbr = [shorten_model_name(m) for m in models_ordered]

    rows = []
    for combo in sorted(combo_stats.keys(), key=sort_combo_key):
        row = {"combo_name": combo}
        for model in models_ordered:
            if model in combo_stats[combo]:
                row[shorten_model_name(model)] = f"{combo_stats[combo][model]:.2f}"
            else:
                row[shorten_model_name(model)] = "-"
        rows.append(row)

    return pd.DataFrame(rows)[["combo_name"] + models_ordered_abbr]


def output_csv_and_tex(dataset, df_out, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{dataset}_{mode}_rate.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"✅ {dataset} CSV 已保存：{csv_path}")

    # 输出 LaTeX
    tex_path = os.path.join(output_dir, f"{dataset}_{mode}_rate.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df_out.to_latex(index=False, caption=f"{datasets_macro[dataset]} {mode} rate"))
    print(f"✅ {dataset} LaTeX 已保存：{tex_path}")


def output_all_datasets_to_latex(datasets, dfs_out, output_dir):
    latex_lines = []
    latex_lines.append("\\begin{table*}[h]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{mode} rate on different datasets: \\smartbugs, \\reposvulcpp, \\reposvulpy, \\primevulc (sampled by {grade})}}".replace("_", " "))
    latex_lines.append("{")
    latex_lines.append("\\scriptsize")
    latex_lines.append("\\setlength\\tabcolsep{3pt}")
    latex_lines.append("\\renewcommand{\\arraystretch}{0.8}")
    latex_lines.append("\\begin{tabular}{l lccccccccccccccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("dataset & series & \\multicolumn{6}{c}{qwen} & \\multicolumn{4}{c}{llama} & \\multicolumn{2}{c}{deepseek} & \\multicolumn{3}{c}{openai} & \\multicolumn{2}{c}{agent} \\\\")
    latex_lines.append("\\cmidrule(lr){3-8} \\cmidrule(lr){9-12} \\cmidrule(lr){13-14} \\cmidrule(lr){15-17} \\cmidrule(lr){18-19}")
    latex_lines.append(" & model & qn-7b & qn-14b & qn-32b & ds-qn-7b & ds-qn-14b & ds-qn-32b & lm-8b & lm-70b & ds-lm-8b & ds-lm-70b & ds-v3 & ds-r1 & gpt-3.5 & gpt-4o & o3-mini & copilot & codex \\\\")


    # 计算每个 series 的列跨度
    model_headers = []
    for series in SERIES_ORDER:
        models = SERIES_GROUPS[series]
        abbrs = [shorten_model_name(m) for m in models]
        model_headers.extend(abbrs)

    for dataset, df_out in zip(datasets, dfs_out):
        latex_lines.append("\\midrule")
        latex_lines.append(f"\\multirow{{12}}{{*}}{{{datasets_macro[dataset]}}}")
        row = {"dataset": datasets_macro[dataset]}
        
        models_ordered = get_sorted_models()
        models_ordered_abbr = [shorten_model_name(m) for m in models_ordered]
        for _, row in df_out.iterrows():
            combo_name = str(row["combo_name"]).replace("combo_", "")
            values = [  "    & " + combo_name] + [f"\\ccell{{{str(row[m])}}}" for m in models_ordered_abbr]
            latex_lines.append(" & ".join(values) + " \\\\")
            if combo_name == "original":
                latex_lines.append("\\cmidrule(lr){2-19}")
        latex_lines.append(" ")
        

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\label{tab:detection-result-all-datasets}")
    latex_lines.append("\\end{table*}")

    tex_path = os.path.join(output_dir, f"all_datasets_{grade}_detection_result.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    print(f"✅ 综合 LaTeX 已保存：{tex_path}")

if __name__ == "__main__":
    for eval_mode in ["exist", "type"]:
        output_dir = os.path.join(f"RQs/RQ6/{mode}_rate_by_dataset", eval_mode)
        dfs_out = []
        for dataset in datasets:
            print(f"Processing {dataset} ({eval_mode})...")
            df_out = compute(dataset, eval_mode)
            if df_out is not None:
                output_csv_and_tex(dataset, df_out, output_dir)
                dfs_out.append(df_out)
        if dfs_out:
            output_all_datasets_to_latex(datasets, dfs_out, output_dir)