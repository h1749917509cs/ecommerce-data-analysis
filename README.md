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
├── code/                       # 代码
│   ├── CDA.ipynb                      # 分析数据
│   ├── data_cleaning.ipynb            # 数据清洗
│   ├── CDA_23301.ipynb                # 分析特定商品的数据
│   ├── data_cleaning_23301.ipynb      # 清洗特定商品的数据
│   └── visualization.ipynb            # 可视化
│   └── analyse_star_product.ipynb     # 明星产品Top3品类下Top5商品的时间序列特征研究
└── conclusion/
    └── 数据分析论文.docx
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

