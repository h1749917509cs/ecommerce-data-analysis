# 电商零售数据分析项目 (E-commerce Data Analysis)

本项目基于 UCI Online Retail 数据集，运用 Python 完成从数据清洗到商业洞察的全流程分析。通过 **RFM 模型**、**K-means 聚类**、**BCG 矩阵**及 **Apriori 关联规则**，挖掘用户价值与商品关联，为电商精准运营提供数据支撑。

## 项目简介：

*   **数据来源**：[UCI Machine Learning Repository - Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
*   **数据背景**：一家在英国注册的非实体店在线零售商，记录了 2010 年 12 月至 2011 年 12 月期间的交易明细，主营全场合礼品销售。
*   **技术栈**：Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, MLxtend)

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
├── output/
│   └── charts/                        # 可视化图表输出
└── conclusion/
    └── 数据分析论文.docx
```
## 核心分析与可视化

1. 销售趋势分析<br>
    1.1. 蓝色“兔子小夜灯”（23084）是爆款，10–11 月在英国圣诞备货季驱动下爆发式增长，11月冲上约 1.2 万件的峰值，是平季水平的 8–10 倍；其余四款全年大多在 2000 件以内平稳波动，整体呈现典型的 B2B 批发“节前集中囤货、平季零星补货”模式。​
   <img width="1286" height="590" alt="家具装饰top5商品_月度表" src="https://github.com/user-attachments/assets/53620e26-07a6-4f50-9c4c-8941d2f59997" />
    1.2. 该类目由购物袋主导，Top5中有4款是JUMBO BAG（大号购物袋），其中蓝色红波点款（85099B）全年领跑并在11月冲上约5500件；整体节奏不是单一圣诞峰，而是“3月—8月—11月”三波备货潮，12月全线下挫。
   <img width="1287" height="590" alt="礼品与纪念品top5商品_月度表" src="https://github.com/user-attachments/assets/557f5272-a800-49e7-b4ff-e1717a5473dc" />
    1.3. 1–5 月四款商品近乎销量为零，需求从 6 月起爬坡、9–11 月集中释放、12 月骤然退潮；混色小鸟挂饰（84879）是最大例外，它全年有销量并在 8 月就冲出约 6400 件的类目年度最高点，50年代圣诞纸链套装（22086）则是最纯正的圣诞款，11 月以约 5900 件登顶。​这五款商品虽然同属圣诞类目，但“圣诞浓度”差异极大，纸链套装和迪斯科球是纯节日商品，蛋糕纸托绑定圣诞烘焙季，而小鸟挂饰和茶烛灯座实际上具备全年通用属性，这也解释了它们曲线形态的根本不同。
   <img width="1285" height="590" alt="圣诞节用品top5商品_月度表" src="https://github.com/user-attachments/assets/fcffe03a-3036-4288-a055-6e573cf254ef" />
   
2. RFM 客户价值分层<br>
基于RFM模型的分层：
<img width="1585" height="639" alt="BCG_client_form" src="https://github.com/user-attachments/assets/00e3815f-fe63-41a1-8b23-fc7925143bf6" />
<br>
基于K-means模型的分层：
<img width="1448" height="443" alt="k-means_client_form" src="https://github.com/user-attachments/assets/6c819ec0-43a7-4aa8-974b-cfa6d3ac3f5c" />
对比K-means聚类（4簇）与RFM分层（6类型）的结果，两者呈现出高度一致的逻辑互补性。K-means的Cluster 3（顶级VIP）和Cluster 2（高价值忠诚）在RFM中主要对应"重要价值客户"；K-means的Cluster 0（流失风险）在RFM中主要对应"重要风险客户"和"重要留存客户"；K-means的Cluster 1（中坚潜力）则大量分布在"常规客户"和"重要发展客户"中。RFM分层通过业务规则提供了更细粒度的语义标签，而K-means通过距离聚类揭示了数据驱动的自然簇结构。两种方法交叉验证，增强了客户分群结论的稳健性。

3. 商品关联规则挖掘 (Apriori)
从Top 20 关联规则总结出 4 条有价值的关联性


