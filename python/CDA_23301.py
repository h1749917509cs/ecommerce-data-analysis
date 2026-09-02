# -*- coding: utf-8 -*-
"""
RFM 用户分层分析 - StockCode 23301
输出: Excel + 柱状图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# ============ 全局设置 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

FILE_PATH = './Data/Online_Retail_23301_Cleaned.xlsx'
OUTPUT_EXCEL = './RFM_23301_User_Segmentation.xlsx'
OUTPUT_PNG = './RFM_23301_Segments_BarChart.png'


def main():
    # ========== 1. 读取数据 ==========
    print("正在读取数据...")
    df = pd.read_excel(FILE_PATH)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # 只取正常订单（排除退货单）
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C', na=False)].copy()
    print(f"有效订单记录: {len(df):,} 条")

    if df.empty:
        print("错误: 没有有效订单数据，程序终止。")
        return

    # ========== 2. 计算 RFM 指标 ==========
    # 使用数据集中最大日期 +1 作为参考日期（更贴近业务实际）
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
    print(f"RFM 参考日期 (Snapshot Date): {snapshot_date.date()}")

    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                    # Frequency
        'TotalAmount': 'sum'                                       # Monetary
    }).reset_index()
    rfm.columns = ['CustomerID', 'R', 'F', 'M']
    print(f"参与分层的客户数: {len(rfm)}")

    # ========== 3. 五分位打分 (1-5 分) ==========
    # R 越小分越高（最近购买越好），F/M 越大分越高
    rfm['R_Score'] = pd.qcut(rfm['R'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['F'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['M'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['RFM_Score'] = (
        rfm['R_Score'].astype(str) + 
        rfm['F_Score'].astype(str) + 
        rfm['M_Score'].astype(str)
    )

    # ========== 4. 用户分层映射 ==========
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

    # ========== 5. 统计各分层人数 ==========
    seg_count = rfm['用户类型'].value_counts()
    print("\n=== 用户分层统计 ===")
    print(seg_count)

    # ========== 6. 保存 Excel ==========
    rfm.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\nRFM 分层结果已保存至: {OUTPUT_EXCEL}")

    # ========== 7. 绘制柱状图 ==========
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(seg_count))
    
    bars = plt.barh(seg_count.index[::-1], seg_count.values[::-1], color=colors[::-1])
    
    # 在柱子右侧标注数值
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 2, bar.get_y() + bar.get_height()/2, 
                 f'{int(width)}', va='center', fontsize=11, fontweight='bold')
    
    plt.title('RFM Customer Tiering Results (StockCode 23301)', fontsize=15, pad=15)
    plt.xlabel('Customer Count')
    plt.ylabel('Customer Type')
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.show()
    print(f"用户分层柱状图已保存至: {OUTPUT_PNG}")


if __name__ == '__main__':
    main()
