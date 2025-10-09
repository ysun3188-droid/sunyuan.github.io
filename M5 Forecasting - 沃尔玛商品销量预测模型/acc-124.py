import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据集
calendar = pd.read_csv('calendar.csv')  # 日历信息数据
sales_train = pd.read_csv('sales_train_validation.csv')  # 销售训练数据
sell_prices = pd.read_csv('sell_prices.csv')  # 商品销售价格数据

# 将销售数据从宽表格式转换为长表格式
sales_train_melted = sales_train.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],  # 保留的主键列
    var_name='d',  # 转换后的日期列名称
    value_name='sales'  # 转换后的销量列名称
)

# 将销售数据与日历数据合并
calendar['date'] = pd.to_datetime(calendar['date'])  # 将日期字符串转换为日期格式
sales_train_melted = sales_train_melted.merge(calendar, on='d', how='left')  # 按日期列合并

# 将销售数据与价格数据合并
sales_train_melted = sales_train_melted.merge(
    sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left'  # 按商店ID、商品ID和周合并
)

# 特征工程 - 创建滞后特征
for lag in [7, 28]:  # 滞后7天和28天
    sales_train_melted[f'sales_lag_{lag}'] = sales_train_melted.groupby('id')['sales'].shift(lag)

# 创建滚动平均特征（过去7天的销量平均值）
sales_train_melted['sales_rolling_mean_7'] = (
    sales_train_melted.groupby('id')['sales']
    .transform(lambda x: x.shift(7).rolling(7).mean())
)

# 编码分类变量为类别类型
categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1']
for col in categorical_cols:
    sales_train_melted[col] = sales_train_melted[col].astype('category')

# 删除由于滞后和滚动特征而产生的缺失值
sales_train_melted = sales_train_melted.dropna()

# 定义特征和目标变量
X = sales_train_melted[
    ['sales_lag_7', 'sales_lag_28', 'sales_rolling_mean_7', 'snap_CA', 'snap_TX', 'snap_WI'] + categorical_cols
]  # 输入特征
y = sales_train_melted['sales']  # 目标变量（销量）

# 按时间顺序划分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 训练 LightGBM 模型
model = LGBMRegressor(objective='regression', n_estimators=1000, learning_rate=0.05)  # 模型参数
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])  # 训练模型并验证

# 对验证集进行预测
y_pred = model.predict(X_valid)

# 计算加权均方根误差
wrmsse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f'WRMSSE: {wrmsse}')

# 对未来28天进行预测
forecast_days = 28  # 预测天数
future_dates = pd.date_range(start=calendar['date'].max() + pd.Timedelta(days=1), periods=forecast_days)

# 准备未来数据并逐天进行预测
future_data = []
last_data = sales_train_melted[sales_train_melted['date'] == sales_train_melted['date'].max()].copy()

for i in range(forecast_days):
    current_date = future_dates[i]
    current_calendar = calendar[calendar['date'] == current_date]
    
    # 如果当前日期不存在于日历中，创建占位数据
    if current_calendar.empty:
        current_calendar = pd.DataFrame({
            'date': [current_date],
            'wday': [0],  # 默认星期几为0
            'snap_CA': [False],  # 默认促销标记为False
            'snap_TX': [False],
            'snap_WI': [False]
        })
    
    # 为当前日期预测销量（使用模型和前一天的数据）
    X_future = last_data.drop(columns=['sales', 'date'])  # 删除目标列和日期列
    y_future = model.predict(X_future)
    
    future_data.append({'date': current_date, 'predicted_sales': y_future[0]})

# 将未来预测结果转换为 DataFrame
forecast_df = pd.DataFrame(future_data)

# 准备提交的预测数据
sample_submission_path = 'sample_submission.csv'
sample_submission = pd.read_csv(sample_submission_path)

# 根据提交格式生成最终结果
def generate_submission(predictions, sample_submission):
    new_cols = [f'F{i}' for i in range(1, 29)]  # 提交所需的列名
    predictions = predictions.rename({k: v for k, v in zip(predictions.columns[1:], new_cols)}, axis=1)
    
    validation = predictions.copy()
    validation = validation[validation.columns[1:29]]
    validation['id'] = predictions['id'].apply(lambda x: x + '_validation')

    evaluation = predictions.copy()
    evaluation = evaluation[evaluation.columns[29:]]
    evaluation['id'] = predictions['id'].apply(lambda x: x + '_evaluation')

    sub_1 = pd.merge(sample_submission.loc[:30489, 'id'], validation, how='left')
    sub_2 = pd.merge(sample_submission.loc[30490:, 'id'], evaluation, how='left')

    return pd.concat([sub_1, sub_2], axis=0)

# 创建最终提交文件
final_submission = generate_submission(forecast_df, sample_submission)

# 保存为CSV文件
submission_file_path = 'final_submission.csv'
final_submission.to_csv(submission_file_path, index=False)

print(f"Submission saved to {submission_file_path}")
