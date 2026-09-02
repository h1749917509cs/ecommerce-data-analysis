import pandas as pd
import numpy as np

# ============================
# 1. 读取数据
# ============================
file_path = './Data/Online Retail.xlsx'
df = pd.read_excel(file_path)

print(f"原始数据形状: {df.shape}")

# ============================
# 2. 清洗列名（只去空白，不改大小写）
# ============================
# 去除首尾空白、替换内部任意空白为单个空格、移除零宽字符
df.columns = (
    df.columns
    .str.strip()
    .str.replace(r'\s+', ' ', regex=True)
    .str.replace(r'[\u200b\u200c\u200d\ufeff]', '', regex=True)  # 零宽字符
)

print("=== 清洗后的列名 ===")
print(df.columns.tolist())

# ============================
# 3. 列名存在性检查
# ============================
required_cols = ['InvoiceNo', 'StockCode', 'Description', 'Quantity',
                 'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"警告: 缺少以下列: {missing}")
else:
    print("所有核心列均已找到！")

# ============================
# 4. 数据类型转换
# ============================
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df['StockCode'] = df['StockCode'].astype(str)

# CustomerID 转为字符串（处理浮点数形式）
df['CustomerID'] = df['CustomerID'].astype(str).str.replace('.0', '', regex=False)
df['CustomerID'] = df['CustomerID'].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

# ============================
# 5. 缺失值处理
# ============================
# Description 缺失：用相同 StockCode 的众数填充，否则标记 Unknown
desc_map = df.groupby('StockCode')['Description'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
)
df['Description'] = df['Description'].fillna(df['StockCode'].map(desc_map))
df['Description'] = df['Description'].fillna('Unknown')

# CustomerID 缺失：删除（用户级分析必需）
before = len(df)
df = df.dropna(subset=['CustomerID'])
print(f"\n删除 CustomerID 缺失行: {before - len(df)} 行")

print("\n=== 清洗后缺失值统计 ===")
print(df.isnull().sum())

# ============================
# 6. 异常值与业务逻辑清洗
# ============================
# 分离退货单（InvoiceNo 以 C 开头）
mask_cancelled = df['InvoiceNo'].str.startswith('C', na=False)
df_cancel = df[mask_cancelled].copy()
df_normal = df[~mask_cancelled].copy()

# 正常订单 Quantity 必须为正数
df_normal = df_normal[df_normal['Quantity'] > 0]

# 合并（如不需要退货数据，可只保留 df_normal）
df = pd.concat([df_normal, df_cancel], ignore_index=True)

# UnitPrice 必须为正数
df = df[df['UnitPrice'] > 0]

# IQR 方法过滤极端高价异常（3倍IQR）
Q1 = df['UnitPrice'].quantile(0.25)
Q3 = df['UnitPrice'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 3 * IQR
df = df[df['UnitPrice'] <= upper_bound]

# ============================
# 7. 重复值处理
# ============================
before_dup = len(df)
df = df.drop_duplicates(keep='first')
print(f"删除重复行: {before_dup - len(df)} 行")

# ============================
# 8. 衍生特征
# ============================
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Hour'] = df['InvoiceDate'].dt.hour
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek  # 0=周一, 6=周日

# ============================
# 9. Country 标准化
# ============================
df['Country'] = df['Country'].str.strip().str.title()

# ============================
# 10. 保存结果
# ============================
print(f"\n最终数据形状: {df.shape}")
print("\n前5行预览:")
print(df.head())

output_path = './Data/Online_Retail_Cleaned.xlsx'
df.to_excel(output_path, index=False)
print(f"\n清洗后的数据已保存至: {output_path}")

# ============================
# 11. 交互式查询：输入 StockCode 返回 Description
# ============================
print("\n" + "=" * 50)
print("  商品描述查询（输入 0 退出）")
print("=" * 50)

while True:
    code = input("\n请输入 StockCode: ").strip()
    
    # 退出指令
    if code == '0':
        print("已退出查询，程序结束。")
        break
    
    # 空输入跳过
    if not code:
        print("输入不能为空，请重新输入。")
        continue
    
    # 查询映射表
    if code in desc_map:
        desc = desc_map[code]
        print(f"StockCode: {code}")
        print(f"Description: {desc}")
    else:
        print(f"未找到 StockCode '{code}'，请确认输入是否正确。")

