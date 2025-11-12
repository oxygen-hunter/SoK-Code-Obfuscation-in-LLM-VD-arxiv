import sys
import os

import numpy as np
from adjustText import adjust_text


# 添加项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.constants import SERIES_GROUPS, REASONING_GROUPS, LLM_ABBR_MAPPING, LLM_SIZE_MAPPING

# # 全局字体大小设置
# plt.rcParams.update({
#     "font.size": 14,
#     "axes.titlesize": 16,
#     "axes.labelsize": 14,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     "legend.fontsize": 12
# })

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

plt.rcParams.update({
    # 全局基础字号（论文里不要太大）
    "font.size": 10,
    # 坐标轴标题 (xlabel, ylabel)
    "axes.labelsize": 10,
    # 坐标轴标题 (plt.title)
    "axes.titlesize": 11,
    # 坐标轴刻度
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    # 图例
    "legend.fontsize": 9,
    # 线宽、坐标轴粗细（论文图更清晰）
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
})

# 字体设置为无衬线，适合英文论文
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = True  # 英文环境正常显示负号

# 颜色设置todo
COLOR_STYLE = {
    'upgrade': 'seagreen',
    'degrade': 'tomato',
    'ori': 'skyblue',
    'obf': 'lightcoral',
    'diff': 'slateblue',
}

HATCH_STYLE = {
    'upgrade': '//',
    'degrade': '\\',
    'ori': 'xx',
    'obf': '',
    'diff': '',
}

FIGURE_SIZE_DUAL = (7, 3)  # 宽, 高
FIGURE_SIZE_TRI = (3.5,2)


csv_path = "RQs/RQ6_score_shift/score_shift_cases.csv"
output_dir = "RQs/RQ6_score_shift/plots"
os.makedirs(output_dir, exist_ok=True)

# 读入数据
df = pd.read_csv(csv_path)

# ---------------------------
# 构造模型到 group 映射
# ---------------------------
model_to_group = {}
for series_name, groups in REASONING_GROUPS.items():
    for g_type, models in groups.items():
        for m in models:
            model_to_group[m] = (series_name, g_type)  # 返回 (系列名, 推理类型)

# ---------------------------
# None 型模型统一放在最右边
# ---------------------------
max_size = max([s for s in LLM_SIZE_MAPPING.values() if s is not None])
none_size = max_size + 10  # None 映射为更靠右的点

def get_model_size(m):
    return LLM_SIZE_MAPPING[m] if LLM_SIZE_MAPPING[m] is not None else none_size

# ---------------------------
for shift in ["upgrade", "degrade"]:
    df_shift = df[df["shift_type"] == shift]

    if df_shift.empty:
        print(f"⚠️ 没有任何 {shift} 样本！")
        continue

    print(f"✅ {shift} 样本数量: {len(df_shift)}")

    # ---------------------------
    # 分析：vuln_type
    # ---------------------------
    vuln_counts = df_shift["vuln_type"].value_counts().sort_values(ascending=False).head(20)

    plt.figure(figsize=FIGURE_SIZE_DUAL)  # 转置后宽 > 高
    bars = plt.bar(
        vuln_counts.index,  # x轴为漏洞类型
        vuln_counts.values,
        color=COLOR_STYLE[shift],
        hatch=HATCH_STYLE[shift],
        edgecolor="black",
    )

    # 在柱子上标数量
    for i, v in enumerate(vuln_counts.values):
        plt.text(i, v + 1, str(v), ha='center', va='bottom')  # 数字在柱子上方

    plt.ylabel(f"{shift} sample count")
    plt.xlabel("vulnerability type")
    plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，防止拥挤
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{shift}_by_vuln_type.pdf"))
    plt.close()

    print("✅ 漏洞类型分布图已生成。")


    # ---------------------------
    # 统计不同 obf combo 导致的{shift}数量
    # ---------------------------
    combo_counts = df_shift["combo_name"].value_counts().sort_index()

    # 绘图
    plt.figure(figsize=FIGURE_SIZE_DUAL)
    bars = plt.bar(
        combo_counts.index, 
        combo_counts.values, 
        color=COLOR_STYLE[shift], 
        hatch=HATCH_STYLE[shift], 
        edgecolor="black"
    )

    # 在柱子上标数量
    for i, v in enumerate(combo_counts.values):
        plt.text(i, v + 0.5, str(v), ha='center')

    plt.xlabel("obfuscation technique")
    plt.ylabel(f"{shift} sample count")
    # plt.title(f"Distribution of {shift} sample on obfuscation technique")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(output_dir, f"{shift}_by_combo.pdf")
    plt.savefig(output_path)
    plt.close()

    print(f"✅ {shift}分析图已保存到：{output_path}")


    # ---------------------------
    # 分析：dataset（语言）
    # ---------------------------
    # 分母：各数据集总样本数
    totals = df.groupby("dataset").size().rename("total")
    # 分子：各数据集该 shift 的样本数
    counts = df_shift.groupby("dataset").size().rename("count")

    # 合并得到 count/total 与 ratio
    stats = totals.to_frame().join(counts, how="left").fillna(0)
    stats["count"] = stats["count"].astype(int)
    stats["ratio"] = stats["count"] / stats["total"]

    # 按比例从大到小排序，并重置索引得到有序的 0..n-1 位置
    stats = stats.sort_values("ratio", ascending=False).reset_index()  # 列包含: dataset, total, count, ratio

    # 将 x 轴分类设为有序类别，确保绘图严格按排序后的顺序
    stats["dataset"] = pd.Categorical(stats["dataset"], categories=stats["dataset"], ordered=True)

    # 画图
    plt.figure(figsize=FIGURE_SIZE_DUAL)
    ax = sns.barplot(data=stats, x="dataset", y="ratio", color=COLOR_STYLE[shift], edgecolor="black", hatch=HATCH_STYLE[shift])

    # 在柱子上标注：count/total (xx.x%)
    for i, row in stats.iterrows():
        ax.text(
            i, row["ratio"] + 0.01,
            f"{row['count']}/{row['total']} ({row['ratio']*100:.1f}%)",
            ha="center", va="bottom", 
        )

    plt.xlabel("Dataset")
    plt.ylabel(f"{shift} ratio")
    plt.ylim(0, 0.20)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{shift}_by_dataset.pdf"))
    plt.close()

    print("✅ 数据集比例图已生成。")





    # ---------------------------
    # 分析：model
    # ---------------------------
    model_counts = df_shift["model"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=FIGURE_SIZE_DUAL)
    bars = plt.bar(
        model_counts.index, 
        model_counts.values, 
        color=COLOR_STYLE[shift], 
        hatch=HATCH_STYLE[shift],
        edgecolor="black"
    )
    # 在柱子上标数量
    for i, v in enumerate(model_counts.values):
        plt.text(i, v + 0.5, str(v), ha='center')

    plt.xlabel("model")
    plt.ylabel(f"{shift} sample count")
    # plt.title(f"Distribution of {shift} sample on models")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{shift}_by_model.pdf"))
    plt.close()

    print("✅ 模型分布图已生成。")

    


    
    # ---------------------------
    # 分析 LOC（改进版）
    # ---------------------------
    plt.figure(figsize=FIGURE_SIZE_TRI)

    # 取两列最小值和最大值
    min_loc = df_shift[["ori_loc", "obf_loc"]].min().min()
    max_loc = df_shift[["ori_loc", "obf_loc"]].max().max()

    # 自动确定主要分布区间 (percentile 1%~99%)
    lower = df_shift[["ori_loc", "obf_loc"]].stack().quantile(0.01)
    upper = df_shift[["ori_loc", "obf_loc"]].stack().quantile(0.99)

    # 绘制直方图 + KDE
    sns.histplot(df_shift["ori_loc"], bins=20, kde=True, color=COLOR_STYLE["ori"], hatch = HATCH_STYLE['ori'], alpha=0.6, label="original LOC")
    sns.histplot(df_shift["obf_loc"], bins=20, kde=True, color=COLOR_STYLE["obf"], hatch = HATCH_STYLE['obf'], alpha=0.6, label="obfuscated LOC")

    # 设置 x 轴显示主要区间，减少空白
    plt.xlim(lower, upper)
    plt.xlabel("Lines of Code (LOC)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{shift}_by_loc.pdf"))
    plt.close()

    # ---------------------------
    # 分析 Complexity
    # ---------------------------
    plt.figure(figsize=FIGURE_SIZE_TRI)

    # 自动确定主要分布区间 (1%~99% 分位)
    lower = df_shift[["ori_complexity", "obf_complexity"]].stack().quantile(0.01)
    upper = df_shift[["ori_complexity", "obf_complexity"]].stack().quantile(0.99)

    sns.histplot(df_shift["ori_complexity"], bins=30, kde=True,
                color=COLOR_STYLE['ori'], hatch = HATCH_STYLE['ori'], alpha=0.6, label="original complexity")
    sns.histplot(df_shift["obf_complexity"], bins=30, kde=True,
                color=COLOR_STYLE['obf'], hatch = HATCH_STYLE['obf'], alpha=0.6, label="obfuscated complexity")

    plt.xlim(lower, upper)
    plt.xlabel("Code Complexity")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{shift}_by_complexity.pdf"))
    plt.close()

    # ---------------------------
    # LOC 差值直方图
    # ---------------------------
    plt.figure(figsize=FIGURE_SIZE_TRI)

    # 自动确定主要分布区间
    loc_diff = df_shift["obf_loc"] - df_shift["ori_loc"]
    lower, upper = loc_diff.quantile([0.01, 0.99])

    sns.histplot(loc_diff, bins=30, kde=True, color=COLOR_STYLE['diff'], hatch = HATCH_STYLE['diff'], alpha=0.6)
    plt.xlim(lower, upper)
    plt.xlabel("LOC Difference (obf - ori)")
    plt.ylabel("Count")
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{shift}_by_loc_diff.pdf"))
    plt.close()

    # ---------------------------
    # Complexity 差值直方图
    # ---------------------------
    plt.figure(figsize=FIGURE_SIZE_TRI)

    complexity_diff = df_shift["obf_complexity"] - df_shift["ori_complexity"]
    lower, upper = complexity_diff.quantile([0.01, 0.99])

    sns.histplot(complexity_diff, bins=30, kde=True, color=COLOR_STYLE['diff'], hatch = HATCH_STYLE['diff'],  alpha=0.6)
    plt.xlim(lower, upper)
    plt.xlabel("Complexity Difference (obf - ori)")
    plt.ylabel("Count")
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{shift}_by_complexity_diff.pdf"))
    plt.close()

    print("✅ LOC 和 Complexity 分析图已生成。")

    # ---------------------------
    # 按模型系列分组，画散点图
    # ---------------------------
    plt.figure(figsize=(6,4))

    series_markers = {
        "qwen": "o",       # 圆点
        "llama": "s",      # 方块
        "deepseek": "D",   # 菱形
        "openai": "^"      # 三角
    }

    series_colors = {
        "qwen": "tab:blue",
        "llama": "tab:green",
        "deepseek": "tab:orange",
        "openai": "tab:red"
    }

    texts = []

    unique_sizes = set()

    for series, models in SERIES_GROUPS.items():
        sizes = []
        counts = []
        labels = []

        for model in models:
            df_model = df_shift[df_shift["model"] == model]
            if df_model.empty:
                continue

            unique_sizes.add(get_model_size(model))
            sizes.append(get_model_size(model))
            counts.append(len(df_model))
            labels.append(LLM_ABBR_MAPPING[model])

        # 画散点
        plt.scatter(
            sizes, counts,
            label=series.capitalize(),
            color=series_colors[series],
            marker=series_markers[series],
            s=60, edgecolor="black", alpha=0.8
        )

        # 加文字
        for x, y, label in zip(sizes, counts, labels):
            texts.append(plt.text(x, y, label, ha="center", va="bottom", rotation=30))

    # 自动调整文字避免重叠
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

    # 原始刻度
    xticks = sorted(list(unique_sizes))
    xticklabels = [str(int(t)) for t in xticks[:-1]] + ["Unknown"]
    plt.xticks(xticks, xticklabels)

    plt.xlabel("Model size (B parameters)")
    plt.ylabel(f"{shift.capitalize()} count")
    plt.title(f"{shift.capitalize()} vs Model Size (by Series)")
    plt.legend(
        title="Series",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),  # 图外右边，纵向居中
        borderaxespad=0
    )
    plt.tight_layout()

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{shift}_by_model_size.pdf"))
    plt.close()

    print("✅ 按模型系列分组的散点图已生成。")

    # # ---------------------------
    # # 按 REASONING_GROUPS 统计
    # # ---------------------------
    # reasoning_stats = []
    # for group_name, models in REASONING_GROUPS.items():
    #     df_group = df_shift[df_shift["model"].isin(models)]
    #     if df_group.empty:
    #         continue

    #     degrade_count = (df_group["shift_type"] == "degrade").sum()
    #     upgrade_count = (df_group["shift_type"] == "upgrade").sum()
    #     other_count = (df_group["shift_type"] == "other").sum()

    #     reasoning_stats.append({
    #         "group": group_name,
    #         "degrade": degrade_count,
    #         "upgrade": upgrade_count,
    #         "other": other_count,
    #         "total": len(df_group)
    #     })

    # df_reasoning_stats = pd.DataFrame(reasoning_stats).sort_values(by="total", ascending=False)
    # # 输出 CSV
    # df_reasoning_stats.to_csv(os.path.join(output_dir, f"{shift}_by_reasoning_group.csv"), index=False)
    # print(df_reasoning_stats)

    # # 绘制条形图
    # ax = df_reasoning_stats.set_index("group")[["degrade", "upgrade"]].plot(
    #     kind="bar", figsize=(6, 5), stacked=False, color=["tomato", "seagreen"], alpha=0.8
    # )
    # plt.title(f"{shift.capitalize()} by Reasoning Group")
    # plt.ylabel("Count")
    # plt.xticks(rotation=0)
    # for container in ax.containers:
    #     ax.bar_label(container, fmt="%d", label_type="edge", fontsize=9)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f"{shift}_by_reasoning_group.pdf"))
    # plt.close()




    #print(f"🎉 {shift} 分析全部完成！")



# ---------------------------
# 统计各系列 degrade / upgrade / other 数量
# ---------------------------
series_stats = []

for series, models in SERIES_GROUPS.items():
    df_series = df[df["model"].isin(models)].copy()
    if df_series.empty:
        continue

    degrade_count = (df_series["shift_type"] == "degrade").sum()
    upgrade_count = (df_series["shift_type"] == "upgrade").sum()
    other_count   = (df_series["shift_type"] == "other").sum()

    series_stats.append({
        "series": series,
        "degrade": degrade_count,
        "upgrade": upgrade_count,
        "other": other_count,
        "total": len(df_series),
        "degrade_average_model": degrade_count / len(models),
        "upgrade_average_model": upgrade_count / len(models),
    })

# 转 DataFrame
df_series_stats = pd.DataFrame(series_stats)
# 按 degrade average 数量降序排序（可选）
df_series_stats = df_series_stats.sort_values(by="degrade_average_model", ascending=False)

# ---------------------------
# 输出 CSV
# ---------------------------
out_path = os.path.join(output_dir, "degrade_upgrade_by_model_series.csv")
df_series_stats.to_csv(out_path, index=False)
print(f"Series stats saved -> {out_path}")
print(df_series_stats)

df_plot = df_series_stats.set_index("series")[["degrade_average_model", "upgrade_average_model"]]
fig, ax = plt.subplots(figsize=(5, 3))


# 计算柱子位置
bar_width = 0.35
x = np.arange(len(df_plot.index))

# 绘制柱子
bars_degrade = ax.bar(
    x - bar_width/2,
    df_plot["degrade_average_model"],
    width=bar_width,
    color=COLOR_STYLE["degrade"],
    hatch=HATCH_STYLE["degrade"],
    edgecolor="black",
    alpha=0.8,
    label="degrade"
)

bars_upgrade = ax.bar(
    x + bar_width/2,
    df_plot["upgrade_average_model"],
    width=bar_width,
    color=COLOR_STYLE["upgrade"],
    hatch=HATCH_STYLE["upgrade"],
    edgecolor="black",
    alpha=0.8,
    label="upgrade"
)

# 设置 x 轴标签
ax.set_xticks(x)
ax.set_xticklabels(df_plot.index, rotation=45, ha="right")
ax.set_xlabel("Model Series")
ax.set_ylabel("Average Count per Model")
#ax.set_title("Series Upgrade/Degrade Statistics")
ax.legend()

ymax = max(df_plot["degrade_average_model"].max(), df_plot["upgrade_average_model"].max())
ax.set_ylim(0, ymax * 1.2)

# 给每个柱子加上数字
for bars in [bars_degrade, bars_upgrade]:
    ax.bar_label(bars, fmt="%d", label_type="edge", padding=5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "degrade_upgrade_by_model_series.pdf"))
plt.close()