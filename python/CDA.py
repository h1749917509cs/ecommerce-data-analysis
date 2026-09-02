import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# ============ 全局设置 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

FILE_PATH = './Data/Online_Retail_Cleaned.xlsx'


def load_data():
    """读取清洗后的数据"""
    print("正在读取数据...")
    df = pd.read_excel(FILE_PATH)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    # 只取正常订单做分析（退货单单独处理，此处排除）
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C', na=False)].copy()
    print(f"数据加载完成，共 {len(df):,} 条记录")
    return df


# ============================================================
# 方法 1：K-means 聚类分析（基于 RFM 特征）
# ============================================================
def kmeans_analysis(df):
    print("\n" + "=" * 60)
    print("【方法1】K-means 聚类分析")
    print("=" * 60)

    # ---------- 1. 构建 RFM 特征 ----------
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)  # 参考日期
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                    # Frequency
        'TotalAmount': 'sum'                                       # Monetary
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    print(f"参与聚类的客户数: {len(rfm)}")

    # ---------- 2. 标准化 ----------
    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    # ---------- 3. 肘部法选择 K ----------
    sse = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        sse.append(km.inertia_)

    plt.figure(figsize=(9, 5))
    plt.plot(list(K_range), sse, marker='o', linewidth=2, color='#2E86AB')
    plt.title('Elbow Method - SSE vs K', fontsize=14)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('SSE (Inertia)')
    plt.xticks(list(K_range))
    plt.tight_layout()
    plt.savefig('./kmeans_elbow.png', dpi=300)
    plt.show()
    print("肘部图已保存: ./kmeans_elbow.png（请观察拐点确定最佳K）")

    # ---------- 4. 自动选择最佳K（肘部法简化版） ----------
    # 计算相邻点斜率差，找拐点
    sse_diff = np.diff(sse)
    k_best = int(np.argmin(np.abs(sse_diff - np.median(sse_diff)))) + 2
    k_best = max(3, min(k_best, 6))  # 限制在合理范围
    print(f"自动建议最佳K = {k_best}（也可手动修改下方 K 值）")

    # ---------- 5. 执行聚类 ----------
    # 如需手动指定，把下面这行改成: K = 4
    # 观测后k = 4 时曲线放缓，所以选择4而不是k_best(6)
    K = 4
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # ---------- 6. 聚类结果可视化 ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [('Recency', 'Frequency'), ('Recency', 'Monetary'), ('Frequency', 'Monetary')]
    for ax, (x, y) in zip(axes, pairs):
        sns.scatterplot(data=rfm, x=x, y=y, hue='Cluster', palette='Set1', ax=ax, s=40, alpha=0.7)
        ax.set_title(f'{x} vs {y}')
    plt.suptitle('K-means Clustering Results (RFM)', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig('./kmeans_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ---------- 7. 输出各簇特征 ----------
    cluster_summary = rfm.groupby('Cluster')[features].mean().round(2)
    cluster_summary['客户数'] = rfm.groupby('Cluster').size()
    print("\n各聚类簇特征（均值）:")
    print(cluster_summary)
    cluster_summary.to_excel('./kmeans_cluster_summary.xlsx')
    rfm.to_excel('./kmeans_rfm_result.xlsx', index=False)
    print("\n结果已保存: kmeans_cluster_summary.xlsx, kmeans_rfm_result.xlsx")


# ============================================================
# 方法 2：分类分析（RFM 模型 + BCG 矩阵）
# ============================================================
def classification_analysis(df):
    print("\n" + "=" * 60)
    print("【方法2】分类分析（RFM 模型 + BCG 矩阵）")
    print("=" * 60)

    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)

    # ============ Part A: 用户 RFM 模型 ============
    print("\n--- Part A: 用户 RFM 模型 ---")
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'R', 'F', 'M']

    # 五分位打分（1-5分），R 越小分越高，F/M 越大分越高
    rfm['R_Score'] = pd.qcut(rfm['R'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['F'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['M'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

    # 用户分层映射
    def rfm_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        if r >= 4 and f >= 4 and m >= 4:
            return 'Important Value Customers'
        elif r >= 4 and f <= 2:
            return 'Important development clients'
        elif r <= 2 and f >= 4 and m >= 4:
            return 'Important retain customers'
        elif r <= 2 and f <= 2:
            return 'Important at-risk customers'
        elif r >= 4 and f >= 2 and m <= 2:
            return 'General Value Customers'
        else:
            return 'Regular Customers'
    rfm['用户类型'] = rfm.apply(rfm_segment, axis=1)

    # 可视化
    plt.figure(figsize=(10, 6))
    seg_count = rfm['用户类型'].value_counts()
    sns.barplot(x=seg_count.values, y=seg_count.index, palette='Set2')
    plt.title('RFM customers segmentation results', fontsize=14)
    plt.xlabel('number of customers')
    plt.ylabel('Customers Type')
    for i, v in enumerate(seg_count.values):
        plt.text(v + 5, i, str(v), va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('./rfm_segments.png', dpi=300)
    plt.show()

    print("RFM 用户分层统计:")
    print(seg_count)
    rfm.to_excel('./rfm_user_classification.xlsx', index=False)
    print("已保存: rfm_user_classification.xlsx")

    # ============ Part B: 商品 BCG 矩阵 ============
    print("\n--- Part B: 商品 BCG 矩阵 ---")
    # 计算每个商品的总销量（份额代理）和增长率（后半年 vs 前半年）
    df['Period'] = np.where(df['InvoiceDate'] >= df['InvoiceDate'].quantile(0.5), 'H2', 'H1')
    bcg = df.groupby(['StockCode', 'Period'])['Quantity'].sum().unstack(fill_value=0)
    if 'H1' not in bcg.columns:
        bcg['H1'] = 0
    if 'H2' not in bcg.columns:
        bcg['H2'] = 0

    # 增长率 = (H2 - H1) / (H1 + 1)  （+1 避免除零）
    bcg['增长率'] = (bcg['H2'] - bcg['H1']) / (bcg['H1'].replace(0, np.nan))
    bcg['增长率'] = bcg['增长率'].fillna(0)
    # 相对市场份额 = 商品总销量 / 最大商品销量
    bcg['总销量'] = bcg['H1'] + bcg['H2']
    bcg['相对份额'] = bcg['总销量'] / bcg['总销量'].max()
    bcg = bcg.reset_index()

    # 用中位数划分高低
    growth_median = bcg['增长率'].median()
    share_median = bcg['相对份额'].median()

    def bcg_classify(row):
        high_growth = row['增长率'] >= growth_median
        high_share = row['相对份额'] >= share_median
        if high_growth and high_share:
            return 'Star'
        elif not high_growth and high_share:
            return 'Cash Cow'
        elif high_growth and not high_share:
            return 'Question'
        else:
            return 'Dog'
    bcg['BCG类型'] = bcg.apply(bcg_classify, axis=1)

    # BCG 散点图（为可视化清晰，取对数缩放）
    plt.figure(figsize=(10, 8))
    colors = {'Star': '#E74C3C', 'Cash Cow': '#27AE60',
              'Question': '#F39C12', 'Dog': '#95A5A6'}
    for cat, color in colors.items():
        sub = bcg[bcg['BCG类型'] == cat]
        plt.scatter(sub['相对份额'], sub['增长率'], s=30, c=color, label=cat, alpha=0.6)
    plt.axvline(share_median, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(growth_median, color='gray', linestyle='--', alpha=0.5)
    plt.xscale('log')
    plt.title('Product BCG Matrix', fontsize=14)
    plt.xlabel('Relative Market Share (log)')
    plt.ylabel('Sales Growth Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./bcg_matrix.png', dpi=300)
    plt.show()

    print("BCG 分类统计:")
    print(bcg['BCG类型'].value_counts())
    bcg.to_excel('./bcg_product_classification.xlsx', index=False)
    print("已保存: bcg_product_classification.xlsx")


# ============================================================
# 方法 3：关联性分析（Apriori，含三指标解读）
# ============================================================
def association_analysis(df):
    print("\n" + "=" * 60)
    print("【方法3】关联性分析（Apriori 算法）")
    print("=" * 60)
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        print("缺少 mlxtend 库，请先安装: pip install mlxtend")
        return

    # ---------- 1. 数据量大，先采样/过滤 ----------
    # 只取英国市场，且过滤掉下单商品种类过少（<=1）的订单
    df_uk = df[df['Country'] == 'United Kingdom'].copy()
    print(f"英国市场订单数: {df_uk['InvoiceNo'].nunique():,}")

    # 构建购物篮：行=InvoiceNo，列=StockCode，值=是否购买
    basket = df_uk.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack(fill_value=0)
    # 转为 one-hot（买过为1）
    basket = (basket > 0).astype(int)
    # 只保留商品数 >=2 的订单（有关联意义）
    basket = basket[basket.sum(axis=1) >= 2]

    # ---------- 2. Apriori 挖掘频繁项集 ----------
    # min_support 可调；数据量大时调高以提高速度
    min_support = 0.02
    print(f"挖掘频繁项集（min_support={min_support}）...")
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    print(f"发现频繁项集: {len(frequent)} 个")
    if frequent.empty:
        print("未发现频繁项集，请降低 min_support 后重试。")
        return

    # ---------- 3. 生成关联规则 ----------
    rules = association_rules(frequent, metric='lift', min_threshold=1)
    rules = rules.sort_values('lift', ascending=False).head(20)
    print(f"生成关联规则: {len(rules)} 条（已按提升度排序取Top20）")

    # ---------- 4. 三指标解读 ----------
    print("\n" + "-" * 50)
    print("【关联规则三指标解读】")
    print("-" * 50)
    print("1. Support（支持度）= 同时包含A和B的交易数 / 总交易数")
    print("   含义: A和B一起出现的普遍程度。值越高越常见。")
    print("2. Confidence（置信度）= P(B|A) = 买A时也买B的概率")
    print("   含义: 买了A后购买B的可靠性。0~1，越高越可靠。")
    print("3. Lift（提升度）= Confidence / P(B) = P(A∩B) / [P(A)×P(B)]")
    print("   含义: 买A对买B的提升程度。")
    print("   >1 正相关（互相促进），=1 独立无关，<1 负相关。")
    print("-" * 50)

    # 展示规则（格式化输出）
    display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    rules_display = rules[display_cols].copy()
    rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
    rules_display['support'] = rules_display['support'].round(4)
    rules_display['confidence'] = rules_display['confidence'].round(4)
    rules_display['lift'] = rules_display['lift'].round(4)
    print("\nTop 20 关联规则:")
    print(rules_display.to_string(index=False))

    rules.to_excel('./association_rules.xlsx', index=False)
    print("\n已保存: association_rules.xlsx")

    # ---------- 5. 可视化 ----------
    plt.figure(figsize=(10, 6))
    plt.scatter(rules['support'], rules['confidence'],
                s=rules['lift'] * 50, c=rules['lift'], cmap='YlOrRd', alpha=0.7, edgecolors='gray')
    plt.colorbar(label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence (bubble size=Lift)', fontsize=13)
    plt.tight_layout()
    plt.savefig('./association_rules_scatter.png', dpi=300)
    plt.show()


# ============================================================
# 主程序：选择性运行
# ============================================================
def main():
    print("=" * 60)
    print("  Online Retail 数据挖掘分析工具")
    print("=" * 60)
    print("请选择要运行的分析（可输入多个，用逗号分隔，如 1,3）:")
    print("  1 - K-means 聚类分析（含肘部图）")
    print("  2 - 分类分析（RFM 模型 + BCG 矩阵）")
    print("  3 - 关联性分析（Apriori，三指标解读）")
    print("  0 - 运行全部")
    print("-" * 60)

    choice = input("请输入选项: ").strip()

    df = None
    need_load = {'1', '2', '3', '0'}
    if any(c in need_load for c in choice.split(',')):
        df = load_data()

    if '0' in choice.split(','):
        kmeans_analysis(df)
        classification_analysis(df)
        association_analysis(df)
    else:
        if '1' in choice.split(','):
            kmeans_analysis(df)
        if '2' in choice.split(','):
            classification_analysis(df)
        if '3' in choice.split(','):
            association_analysis(df)

    print("\n分析完成！")


if __name__ == '__main__':
    main()
