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

3. 商品关联规则挖掘 (Apriori)<br>
从Top 20 关联规则总结出 2 条有价值的关联性<br>
    3.1. 摄政茶杯和碟组合（颜色：绿色22697，粉色22698，玫瑰22699）<br>
        3.1.1. 组合占据绝对主导地位, 这12条规则均围绕 {22697, 22698, 22699} 三件商品展开，占据了 Top 20 规则的 60%。该组合的支撑力和方向一致性极强，无论从哪个方向切入，提升度均超过 17。<br>
        3.1.2. 核心推荐规则表现突出, 最佳推荐效果：以 {22699, 22697} 同时出现时推荐 22698 的效果最为突出，置信度达到 71.6%，提升度高达 22.45。最强单向规则：{22699, 22698} → {22697} 的置信度高达 89.0%。这意味着，在已购买 22699 和 22698 两件商品的客户中，近九成会同时购买 22697。<br>
        3.1.3. 运营策略建议, 基于上述极强的关联性，该三件商品非常适合以 "套装捆绑" 的方式进行统一陈列和定价，以最大化客单价与转化率。
   <img width="508" height="523" alt="group_1" src="https://github.com/user-attachments/assets/31dc47b4-45a0-4d75-8a61-ab2b74e0a003" />
   <br>
   3.2. 园艺跪垫组合（图案：一杯茶23300，保持冷静23301）<br>
       3.2.1. 这对商品呈现出高度对称的关联关系，两个方向的提升度均高达 20.30。无论购买哪一方，都会显著提升对方被同时加入购物车的概率。<br>
       3.2.2. 鉴于两者极强的双向绑定属性，建议将 22629 与 22630 作为 "双子星组合" 进行强绑定营销。在商品详情页设置 "搭配购买" 模块，或在购物车环节进行交叉推荐，利用其高置信度特性有效提升连带率（Uptake Rate）。
   <img width="444" height="123" alt="group_2" src="https://github.com/user-attachments/assets/9ddd27e1-02c5-48b6-9590-cfcc991112db" />
   <br>

4. 商品分类 (BCG Matrix)<br>
基于全量 SKU 的**相对市场份额**（Relative Market Share，对数刻度）与**销售增长率**（Sales Growth Rate）构建 BCG 矩阵，将商品划分为明星、现金牛、问题、瘦狗四类，用于指导库存、采购与营销资源的差异化分配。
<img width="989" height="790" alt="Product_BCG_Matrix" src="https://github.com/user-attachments/assets/be8a9124-1104-4e46-8b48-0ff2072367ac" />
<br>
从整体分布看，四类商品呈现明显的"两极分化"：高份额商品密集贴附于零增长线附近，而低份额区域则聚集了大量近乎零动销的长尾 SKU。

各象限明细：
现金牛产品（Cash Cow）
<img width="947" height="169" alt="image" src="https://github.com/user-attachments/assets/3e6d7e06-2d43-4ac7-88bf-295d3bbe09fd" />
**策略建议**：维持库存深度、保障供应稳定、适度控制营销投入，最大化利润贡献。
<br>
明星产品（Star)
<img width="937" height="169" alt="image" src="https://github.com/user-attachments/assets/a89c38f2-a592-47ee-bfbf-c202250a86ec" />
**策略建议**：确保库存充足、扩大采购规模、给予首页推荐或捆绑引流，推动其向现金牛演进。
<br>
问题产品（Question）
<img width="799" height="169" alt="image" src="https://github.com/user-attachments/assets/c63b314f-82ec-4e06-804c-9abb57361dd0" />
**策略建议**：甄别真正具有品类创新价值的单品进行小规模试销培育；对缺乏延展性的长尾产品避免盲目投入。
<br>
瘦狗产品（Dog）
<img width="853" height="169" alt="image" src="https://github.com/user-attachments/assets/8a479e83-d5d9-4f7a-9038-0777de648423" />
**策略建议**：长期零动销商品果断下架；偶发性需求商品转为"以销定采"预售模式，释放运营资源。
<br>
关键洞察:
1. **利润高度集中**：21212 单品相对份额达 0.416、销量 3.4 万件，头部现金牛贡献了主要销量，是供应链稳定性的核心保障对象。
2. **季节性增长显著**：明星产品中圣诞主题商品（20668、16169E）增长强劲，需提前规划季节性备货与流量节奏。
3. **警惕低基数陷阱**：问题产品的高增长率（如 21261 达 2,550%）多源于极低基数，绝对销量有限，应以小步试销验证后再追加投入。
4. **长尾包袱沉重**：瘦狗象限 SKU 数量庞大但单量极低（如 10123C 总销量仅 5 件），建议分批清理，将仓储与运营资源让渡给头部商品。 











