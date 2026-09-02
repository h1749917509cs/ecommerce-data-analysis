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
# 5. 筛选目标商品 StockCode = 23301
# ============================
TARGET_CODE = '23301'
before_filter = len(df)
df = df[df['StockCode'] == TARGET_CODE].copy()
print(f"筛选 StockCode = {TARGET_CODE}: {before_filter} → {len(df)} 行")

if df.empty:
    print(f"错误: 未找到 StockCode = {TARGET_CODE} 的记录，请确认编码是否正确。")
    exit()

# ============================
# 6. 缺失值处理
# ============================
# Description 缺失：用相同 StockCode 的众数填充，否则标记 Unknown
desc_map = df.groupby('StockCode')['Description'].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
)
df['Description'] = df['Description'].fillna(df['StockCode'].map(desc_map))
df['Description'] = df['Description'].fillna('Unknown')

# CustomerID 缺失：删除
before = len(df)
df = df.dropna(subset=['CustomerID'])
print(f"删除 CustomerID 缺失行: {before - len(df)} 行")

print("=== 清洗后缺失值统计 ===")
print(df.isnull().sum())

# ============================
# 7. 异常值与业务逻辑清洗
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

# ============================
# 8. 重复值处理
# ============================
before_dup = len(df)
df = df.drop_duplicates(keep='first')
print(f"删除重复行: {before_dup - len(df)} 行")

# ============================
# 9. 衍生特征
# ============================
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['Hour'] = df['InvoiceDate'].dt.hour
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek  # 0=周一, 6=周日

# ============================
# 10. Country 标准化
# ============================
df['Country'] = df['Country'].str.strip().str.title()

# ============================
# 11. 保存结果
# ============================
print(f"最终数据形状: {df.shape}")
print("前5行预览:")
print(df.head())

output_path = './Data/Online_Retail_23301_Cleaned.xlsx'
df.to_excel(output_path, index=False)
print(f"清洗后的数据已保存至: {output_path}")