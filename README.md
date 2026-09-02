# ecommerce-data-analysis

## 项目简介

本项目基于 UCI Online Retail 数据集，运用 Python 完成从数据清洗到商业洞察的全流程数据分析，通过 RFM 客户价值分析、K-means 聚类、BCG 矩阵商品分类及 Apriori 关联规则挖掘，为电商精准运营提供数据支撑。

### 数据来源：

[Online Retail - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail)  ，数据集来源于一家在英国注册的非实体店在线零售商，记录 2010 年 12 月至 2011 年 9 月期间约十个月的全部交易明细，主营全场合礼品销售。

## 技术栈

Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)



## 项目结构
```text
ecommerce-data-analysis/
├── README.md
├── python/                       # python 分析代码
│   ├── CDA.py                      # 分析数据
│   ├── data_cleaning.py            # 数据清洗
│   ├── CDA_23301.py                # 分析特定时间的数据
│   ├── data_cleaning_23301.py      # 清洗特定时间的数据
│   └── visualization.py            # 可视化
└── output/
    └── charts/                     # 可视化图表
```
## 分析内容

1. 数据清洗与预处理
2. 探索性数据分析（EDA）
3. K-means 客户聚类
4. RFM 客户价值分析
5. BCG矩阵商品分类
6. Apriori 关联规则挖掘

## 核心发现与结论

- 发现 4 个高价值客户群体，其中“高价值忠诚客户”占4.4%，其贡献额了远超其余 95% 的客户
- 关联规则挖掘得出 6 条有效规则，其中置信度最高的规则为“购买茶杯图案的园艺跪垫的用户有 73.2% 概率同时购买KEEP CALM图案的园艺跪垫
- 可视化看板见 output/charts/ 目录

