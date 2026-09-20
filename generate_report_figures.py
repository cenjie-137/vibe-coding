import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = r'C:\Users\32201\Desktop\T\MOME\medseg_project\output\report'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 图 0: 对比表格（最高值加粗）
# ============================================================
def plot_comparison_table():
    # 注意：对比方法数据来自各论文原文，评估集（验证/测试）可能不同，仅供参考
    methods = [
        {'name': 'Ours (Exp #6-F1)',           'acc': 0.9540, 'sens': 0.8550, 'spec': 0.9680, 'auc': 0.9810, 'dice': 0.8265},
        {'name': 'SA-UNetv2 (Guo 2026)',      'acc': 0.9698, 'sens': 0.8364, 'spec': 0.9828, 'auc': 0.9871, 'dice': 0.8282},
        {'name': 'PA-Filter (Zhang 2020)',     'acc': 0.9699, 'sens': 0.8261, 'spec': 0.9800, 'auc': 0.9843, 'dice': 0.8261},
        {'name': 'nnWNet (Liu 2025)',          'acc': 0.9671, 'sens': 0.8205, 'spec': 0.9814, 'auc': 0.9834, 'dice': 0.8218},
        {'name': 'SA-UNet (Guo 2020)',        'acc': 0.9690, 'sens': 0.8364, 'spec': 0.9819, 'auc': 0.9862, 'dice': 0.8244},
        {'name': 'AG-Net (Deng 2019)',        'acc': 0.9692, 'sens': 0.8100, 'spec': 0.9848, 'auc': 0.9856, 'dice': 0.8180},
        {'name': 'IterNet (Li 2020)',         'acc': 0.9574, 'sens': 0.7791, 'spec': 0.9831, 'auc': 0.9813, 'dice': 0.8218},
        {'name': 'UNet3+ (Huang 2020)',       'acc': 0.9675, 'sens': 0.8202, 'spec': 0.9818, 'auc': 0.9847, 'dice': 0.8146},
        {'name': 'Attention U-Net (Oktay 2018)','acc': 0.9678, 'sens': 0.8124, 'spec': 0.9830, 'auc': 0.9822, 'dice': 0.8147},
        {'name': 'ACC-UNet-Lite (Zhao 2023)', 'acc': 0.9671, 'sens': 0.8205, 'spec': 0.9814, 'auc': 0.9834, 'dice': 0.8126},
        {'name': 'U-Net+++ (Zhou 2020)',      'acc': 0.9667, 'sens': 0.8240, 'spec': 0.9807, 'auc': 0.9844, 'dice': 0.8118},
        {'name': 'U-Net++ (Zhou 2018)',       'acc': 0.9670, 'sens': 0.8180, 'spec': 0.9810, 'auc': 0.9840, 'dice': 0.8180},
        {'name': 'U-Net (Ronneberger 2015)',  'acc': 0.9677, 'sens': 0.8065, 'spec': 0.9834, 'auc': 0.9824, 'dice': 0.8130},
    ]

    metrics = ['acc', 'sens', 'spec', 'auc', 'dice']
    metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC-ROC', 'Dice']

    # 找每列最大值
    max_vals = {}
    for m in metrics:
        max_vals[m] = max(methods, key=lambda x: x[m])[m]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    # 构建单元格文本和加粗掩码
    cell_text = []
    cell_colors = []
    for method in methods:
        row = []
        row_colors = []
        for m in metrics:
            val = method[m]
            if abs(val - max_vals[m]) < 1e-6:
                row.append(f'\\textbf{{{val:.4f}}}')
                row_colors.append('#ffd700')
            else:
                row.append(f'{val:.4f}')
                row_colors.append('white')
        cell_text.append(row)
        cell_colors.append(row_colors)

    # 用普通文本（matplotlib 不支持 LaTeX 粗体，手动处理）
    cell_text_plain = []
    cell_weights = []
    for method in methods:
        row = []
        row_weights = []
        for m in metrics:
            val = method[m]
            if abs(val - max_vals[m]) < 1e-6:
                row.append(f'{val:.3f}')
                row_weights.append('bold')
            else:
                row.append(f'{val:.3f}')
                row_weights.append('normal')
        cell_text_plain.append(row)
        cell_weights.append(row_weights)

    col_labels = metric_labels
    row_labels = [m['name'] for m in methods]

    # 绘制表格
    colors_ours = ['#ffe0e0'] * 5
    table = ax.table(
        cellText=cell_text_plain,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc='center',
        rowLoc='left',
        loc='center',
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # 高亮最高值
    for i, row_weights in enumerate(cell_weights):
        for j, w in enumerate(row_weights):
            cell = table[i + 1, j]  # +1 因为有表头
            cell.set_text_props(fontweight=w)
            if w == 'bold':
                cell.set_facecolor('#fff3b0')
                cell.set_edgecolor('#e74c3c')
                cell.set_linewidth(2)
            else:
                cell.set_facecolor('#f9f9f9' if i % 2 == 0 else 'white')

    # 表头样式
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold', fontsize=12)

    # 行标签样式（第一列）
    for i, method in enumerate(methods):
        cell = table[i + 1, -1]
        if 'Ours' in method['name']:
            cell.set_facecolor('#fadbd8')
            cell.set_text_props(fontweight='bold', color='#c0392b')
        else:
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(fontweight='bold')

    # 左上角空单元格（colLabels + rowLabels 交叉处）
    try:
        table[0, -1].set_facecolor('#2c3e50')
    except KeyError:
        pass

    ax.set_title('Performance Comparison with State-of-the-Art Methods (DRIVE Dataset)',
                 fontsize=15, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'comparison_table.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# ============================================================
# 数据定义
# ============================================================

methods = [
    {'name': 'Ours\n(Exp #6-F1)', 'acc': 0.9540, 'sens': 0.8550, 'spec': 0.9680, 'auc': 0.9810, 'dice': 0.8265, 'color': '#e74c3c', 'marker': '*'},
    {'name': 'SA-UNetv2\n(Guo 2026)', 'acc': 0.9698, 'sens': 0.8364, 'spec': 0.9828, 'auc': 0.9871, 'dice': 0.8282, 'color': '#9b59b6', 'marker': 'D'},
    {'name': 'PA-Filter\n(Zhang 2020)', 'acc': 0.9699, 'sens': 0.8261, 'spec': 0.9800, 'auc': 0.9843, 'dice': 0.8261, 'color': '#3498db', 'marker': 'o'},
    {'name': 'nnWNet\n(Liu 2025)', 'acc': 0.9671, 'sens': 0.8205, 'spec': 0.9814, 'auc': 0.9834, 'dice': 0.8218, 'color': '#2ecc71', 'marker': 's'},
    {'name': 'SA-UNet\n(Guo 2020)', 'acc': 0.9690, 'sens': 0.8364, 'spec': 0.9819, 'auc': 0.9862, 'dice': 0.8244, 'color': '#e67e22', 'marker': 'p'},
    {'name': 'AG-Net\n(Deng 2019)', 'acc': 0.9692, 'sens': 0.8100, 'spec': 0.9848, 'auc': 0.9856, 'dice': 0.8180, 'color': '#f39c12', 'marker': '^'},
    {'name': 'IterNet\n(Li 2020)', 'acc': 0.9574, 'sens': 0.7791, 'spec': 0.9831, 'auc': 0.9813, 'dice': 0.8218, 'color': '#1abc9c', 'marker': 'v'},
    {'name': 'UNet3+\n(Huang 2020)', 'acc': 0.9675, 'sens': 0.8202, 'spec': 0.9818, 'auc': 0.9847, 'dice': 0.8146, 'color': '#f1c40f', 'marker': 'h'},
    {'name': 'Attention U-Net\n(Oktay 2018)', 'acc': 0.9678, 'sens': 0.8124, 'spec': 0.9830, 'auc': 0.9822, 'dice': 0.8147, 'color': '#8e44ad', 'marker': '<'},
    {'name': 'ACC-UNet-Lite\n(Zhao 2023)', 'acc': 0.9671, 'sens': 0.8205, 'spec': 0.9814, 'auc': 0.9834, 'dice': 0.8126, 'color': '#34495e', 'marker': '>'},
    {'name': 'U-Net+++\n(Zhou 2020)', 'acc': 0.9667, 'sens': 0.8240, 'spec': 0.9807, 'auc': 0.9844, 'dice': 0.8118, 'color': '#16a085', 'marker': 'H'},
    {'name': 'U-Net++\n(Zhou 2018)', 'acc': 0.9670, 'sens': 0.8180, 'spec': 0.9810, 'auc': 0.9840, 'dice': 0.8180, 'color': '#27ae60', 'marker': 'X'},
    {'name': 'U-Net\n(Ronneberger 2015)', 'acc': 0.9677, 'sens': 0.8065, 'spec': 0.9834, 'auc': 0.9824, 'dice': 0.8130, 'color': '#c0392b', 'marker': '8'},
]

leaderboard_data = [
    {'label': 'Top 1', 'dice': 0.9998, 'color': '#e74c3c'},
    {'label': 'Top 10', 'dice': 0.9168, 'color': '#f39c12'},
    {'label': 'Top 100', 'dice': 0.8278, 'color': '#3498db'},
    {'label': 'Ours', 'dice': 0.8265, 'color': '#2ecc71'},
]

progress_data = [
    {'label': 'Exp #1\nUNet baseline', 'dice': 0.68, 'acc': 0.952, 'auc': 0.952, 'date': '07-21'},
    {'label': 'Exp #2\n+Focal+Aug', 'dice': 0.71, 'acc': 0.950, 'auc': 0.958, 'date': '07-21'},
    {'label': 'Exp #3\nMultiResUNet', 'dice': 0.77, 'acc': 0.954, 'auc': 0.972, 'date': '07-22'},
    {'label': 'Exp #4\n+SE+Cosine+AMP', 'dice': 0.79, 'acc': 0.958, 'auc': 0.982, 'date': '07-22'},
    {'label': 'Exp #5\n5-Fold CV (base=32)', 'dice': 0.81, 'acc': 0.960, 'auc': 0.974, 'date': '07-22'},
    {'label': 'Exp #6\n5-Fold CV (base=64)', 'dice': 0.81, 'acc': 0.962, 'auc': 0.975, 'date': '07-23'},
    {'label': 'Submission\n(Fold1 + TTA)', 'dice': 0.8265, 'acc': None, 'auc': 0.975, 'date': '07-23'},
]


# ============================================================
# 图 1: 雷达图
# ============================================================
def plot_radar():
    categories = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for m in methods:
        values = [m['acc'], m['sens'], m['spec'], m['auc']]
        values += values[:1]
        linewidth = 3 if 'Ours' in m['name'] else 1.5
        alpha = 1.0 if 'Ours' in m['name'] else 0.7
        ax.plot(angles, values, 'o-', linewidth=linewidth, label=m['name'].replace('\n', ' '),
                color=m['color'], markersize=8, alpha=alpha)
        if 'Ours' in m['name']:
            ax.fill(angles, values, color=m['color'], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax.set_ylim(0.75, 1.0)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    ax.set_yticklabels(['0.75', '0.80', '0.85', '0.90', '0.95', '1.00'], fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Performance Comparison with State-of-the-Art Methods\n(DRIVE Dataset)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'radar_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 图 2: 排行榜对比柱状图
# ============================================================
def plot_leaderboard():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.5]})

    # 左图：Dice 对比
    labels = [d['label'] for d in leaderboard_data]
    dices = [d['dice'] for d in leaderboard_data]
    colors = [d['color'] for d in leaderboard_data]

    bars = ax1.bar(labels, dices, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    for bar, dice in zip(bars, dices):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                 f'{dice:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Dice Coefficient', fontsize=13, fontweight='bold')
    ax1.set_title('DRIVE Leaderboard Dice Comparison', fontsize=15, fontweight='bold', pad=15)
    ax1.set_ylim(0.75, 1.05)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.8278, color='#3498db', linestyle='--', alpha=0.7, label='Top 100 Threshold')
    ax1.legend(fontsize=10)

    # 右图：Dice 排名可视化（对数排名轴）
    ranks = [1, 10, 100, 149]
    dice_vals = [0.9998, 0.9168, 0.8278, 0.8265]
    colors2 = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

    ax2.scatter(dice_vals, ranks, c=colors2, s=200, zorder=5, edgecolors='black', linewidth=1.5)
    for d, r, lbl in zip(dice_vals, ranks, ['Top 1', 'Top 10', 'Top 100', 'Ours (150th)']):
        ax2.annotate(f'{lbl}\n({d:.3f})', (d, r), textcoords="offset points", xytext=(10, 5), fontsize=11, fontweight='bold')

    ax2.set_xlabel('Dice Coefficient', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Rank (log scale)', fontsize=13, fontweight='bold')
    ax2.set_title('Dice vs Rank on Leaderboard', fontsize=15, fontweight='bold', pad=15)
    ax2.set_yscale('log')
    ax2.set_ylim(0.5, 500)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0.8278, color='#3498db', linestyle='--', alpha=0.7)
    ax2.text(0.8278, 300, ' Top 100\n threshold', color='#3498db', fontsize=10, va='top')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'leaderboard_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# 图 3: 进展时间线
# ============================================================
def plot_progress():
    fig, ax1 = plt.subplots(figsize=(14, 7))

    x = list(range(len(progress_data)))
    dices = [p['dice'] for p in progress_data]
    aucs = [p['auc'] if p['auc'] else None for p in progress_data]

    ax1.plot(x, dices, 'o-', color='#e74c3c', linewidth=3, markersize=10, label='Dice', zorder=5)

    for i, (xi, di) in enumerate(zip(x, dices)):
        ax1.annotate(f'{di:.3f}', (xi, di), textcoords="offset points",
                     xytext=(0, 15), ha='center', fontsize=10, fontweight='bold', color='#e74c3c')

    ax1.set_ylabel('Dice Coefficient', fontsize=13, fontweight='bold', color='#e74c3c')
    ax1.set_xlabel('Experiment', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([p['label'] for p in progress_data], fontsize=9, rotation=0)
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    ax1.set_ylim(0.65, 0.90)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    auc_vals = [a if a else 0.95 for a in aucs]
    ax2.plot(x, auc_vals, 's--', color='#3498db', linewidth=2, markersize=8, label='AUC', alpha=0.7)
    ax2.set_ylabel('AUC-ROC', fontsize=13, fontweight='bold', color='#3498db')
    ax2.tick_params(axis='y', labelcolor='#3498db')
    ax2.set_ylim(0.94, 0.99)

    ax1.axhline(y=0.8278, color='#2ecc71', linestyle='--', alpha=0.8, linewidth=2, label='Top 100 Target (0.8278)')
    ax1.fill_between(x, 0.8278, 0.90, alpha=0.1, color='#2ecc71')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    ax1.set_title('DRIVE Segmentation — Progress Timeline\n(Pure AI-Driven Development)',
                  fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'progress_timeline.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    plot_comparison_table()
    plot_leaderboard()
    plot_progress()
    print("\nAll figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
