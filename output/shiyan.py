import matplotlib.pyplot as plt
import numpy as np

# ================= 1. 数据准备 =================
labels = [
    'LearnedRewrite', 
    'LLM-R2\n(Claude-4.5)', 
    'LLM-R2\n(DS-R1)', 
    'R-Bot\n(Claude-4.5)', 
    'R-Bot\n(DS-R1)', 
    'QUITE', 
    'AgentRewrite'
]

# 等价率数据 (Equivalence Rate)
tpch_eq = [93.65, 93.65, 93.65, 98.41, 96.23, 98.00, 100.00]
dsb_eq = [73.71, 85.89, 85.25, 82.05, 83.30, 95.51, 96.79]
calcite_eq = [81.03, 82.75, 82.75, 87.93, 86.20, 91.37, 96.55]

# 改进率数据 (Improvement Rate)
tpch_imp = [33.30, 23.80, 26.99, 41.27, 36.50, 39.68, 42.85]
dsb_imp = [19.87, 29.49, 25.64, 24.00, 22.44, 35.90, 38.46]
calcite_imp = [20.69, 27.59, 26.98, 27.59, 31.03, 34.48, 43.10]

# ================= 2. 全局样式设置 (使用 Clay and Sea 配色) =================
# 提取自 "Clay and Sea" 色卡
colors = ['#C75C3B', '#E0B67D', '#108B8B']  # 深陶土红, 海沙暖黄, 深海蓝绿

# 进一步加大各项字体，尤其是横纵坐标
plt.rcParams.update({
    'font.size': 18,             # 全局基准字体（调大）
    'axes.labelsize': 26,        # X轴/Y轴标题字体（显著调大）
    'axes.titlesize': 24,        # 图表主标题字体
    'xtick.labelsize': 20,       # X轴刻度标签字体（如模型名称，调大）
    'ytick.labelsize': 20,       # Y轴刻度标签字体（如数字比例，调大）
    'legend.fontsize': 18,       # 图例字体（略微调大以匹配整体）
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans']
})

# ================= 3. 绘图函数 =================
def plot_bar_chart(data1, data2, data3, ylabel, y_lim, filename):
    x = np.arange(len(labels))
    width = 0.25  # 柱子宽度

    # 加宽图表尺寸以适应更大的字体
    fig, ax = plt.subplots(figsize=(16, 8))

    # 绘制柱状图 (边缘添加极细的深色边框增加立体感)
    rects1 = ax.bar(x - width*1.05, data1, width, label='TPCH', color=colors[0], edgecolor='none', zorder=3)
    rects2 = ax.bar(x, data2, width, label='DSB', color=colors[1], edgecolor='none', zorder=3)
    rects3 = ax.bar(x + width*1.05, data3, width, label='Calcite', color=colors[2], edgecolor='none', zorder=3)

    # 设置标签和边界
    ax.set_ylabel(ylabel, labelpad=25)  # 稍微拉开标题与Y轴数字的距离
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(y_lim)

    # 隐藏上方和右侧的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#555555')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_color('#555555')
    ax.spines['bottom'].set_linewidth(1.5)

    # 添加浅色水平网格线
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.3, zorder=0)

    # 图例设置 (放置在顶部中间，去除边框)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, handletextpad=0.5, columnspacing=2.0)

    # 在柱子上添加数据标签 (加大数字字体)
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            text_str = f'{height:g}' 
            ax.annotate(text_str,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 垂直偏移加大
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14, color='#333333', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # 调整布局并保存
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# ================= 4. 生成两张图表 =================
# 绘制 Equivalence Rate 图
plot_bar_chart(tpch_eq, dsb_eq, calcite_eq, 'Equivalence Rate (%)', [70, 108], 'equivalence_rate_clay_sea.pdf')

# 绘制 Improvement Rate 图
plot_bar_chart(tpch_imp, dsb_imp, calcite_imp, 'Improvement Rate (%)', [0, 52], 'improvement_rate_clay_sea.pdf')