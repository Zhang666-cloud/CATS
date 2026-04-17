import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib

# 设置全局字体为Times New Roman，更适合学术论文
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['mathtext.fontset'] = 'stix'  # 数学字体

# 1. 从文件中读取PLPD数据
def load_plpd_data(file_path):
    with open(file_path, 'rb') as f:
        plpd_data = pickle.load(f)
    return plpd_data

# 2. 准备数据
plpd_file_path = '/mnt/sda/PythonProject/ZYF_projects/ATP-master/save_plpd/avg_len_4_plpd_values_PACS2.pkl'
plpd_data = load_plpd_data(plpd_file_path)

# 指定类别和对应的CLPD值
categories = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

# 假设您的PLPD数据是按照这些类别顺序存储的
# 如果数据存储方式不同，您可能需要调整这部分代码
plpd_values = [plpd_data.get(i, 0) for i in range(len(categories))]

# 3. 创建高质量图表
plt.figure(figsize=(10, 6), dpi=300)  # 高分辨率

# 使用科学配色方案
colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(categories)))

# 绘制柱状图
bars = plt.bar(categories, plpd_values, color=colors, edgecolor='black', linewidth=0.7, alpha=0.9)

# 4. 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,  # 稍微上移避免重叠
             f'{height:.4f}',  # 保留四位小数
             ha='center', va='bottom', fontsize=12, color='black', weight='bold')

# 5. 设置坐标轴和标题
plt.ylim(min(plpd_values) - 0.02, max(plpd_values) + 0.05)  # 基于数据的动态范围
plt.xticks(rotation=45, ha='right', fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Category', fontsize=18, labelpad=10)
plt.ylabel('Average CAIN Value', fontsize=18, labelpad=10)
plt.title('CAIN Distribution Across Selected Classes (PACS)',
          fontsize=18, pad=15)

# 6. 添加网格线和背景
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.gca().set_facecolor('#f8f9fa')  # 浅灰色背景提高可读性

# 7. 调整布局并保存
plt.tight_layout()
plt.savefig('pacs_class_clpd_comparison.png', format='png', bbox_inches='tight', dpi=300)
plt.savefig('pacs_class_clpd_comparison.pdf', format='pdf', bbox_inches='tight')  # 矢量图
plt.savefig('pacs_class_clpd_comparison.eps', format='eps', bbox_inches='tight')  # 出版质量

# 显示图表
plt.show()

print(f"图表已保存为: pacs_class_clpd_comparison.png/pdf/eps")