import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据集
calendar = pd.read_csv('calendar.csv')  # 日历数据
sales_train = pd.read_csv('sales_train_validation.csv')  # 销售数据
sell_prices = pd.read_csv('sell_prices.csv')  # 销售价格数据

# 将销售数据从宽表格式转换为长表格式
sales_train_melted = sales_train.melt(
    id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],  # 保留的标识列
    var_name='d',  # 将日期列重命名为 'd'
    value_name='sales'  # 销量列重命名为 'sales'
)

# 将销售数据与日历数据合并
calendar['date'] = pd.to_datetime(calendar['date'])  # 将日历中的日期转换为日期格式
sales_train_melted = sales_train_melted.merge(calendar, on='d', how='left')  # 按 'd' 列（日期）进行合并

# 将销售数据与价格数据合并
sales_train_melted = sales_train_melted.merge(
    sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left'  # 按商店ID、商品ID和周合并
)

# 特征工程 - 创建滞后特征
for lag in [7, 28]:  # 生成滞后7天和28天的销量特征
    sales_train_melted[f'sales_lag_{lag}'] = sales_train_melted.groupby('id')['sales'].shift(lag)

# 创建7天滚动平均特征
sales_train_melted['sales_rolling_mean_7'] = (
    sales_train_melted.groupby('id')['sales']
    .transform(lambda x: x.shift(7).rolling(7).mean())  # 计算过去7天的滚动平均销量
)

# 编码分类变量
categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1']
for col in categorical_cols:
    sales_train_melted[col] = sales_train_melted[col].astype('category')  # 将分类变量转换为类别类型

# 删除由于滞后和滚动平均生成的缺失值
sales_train_melted = sales_train_melted.dropna()

# 定义特征和目标变量
X = sales_train_melted[
    ['sales_lag_7', 'sales_lag_28', 'sales_rolling_mean_7', 'snap_CA', 'snap_TX', 'snap_WI'] + categorical_cols
]  # 特征变量
y = sales_train_melted['sales']  # 目标变量（销量）

# 按时间顺序划分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 为每个分位数（50%、67%、95%、99%）训练模型
quantiles = [0.5, 0.67, 0.95, 0.99]  # 定义分位数
models = {}

# 为每个分位数训练LightGBM回归模型
for quantile in quantiles:
    model_quantile = LGBMRegressor(objective='quantile', alpha=quantile, n_estimators=1000, learning_rate=0.05)  # 设置分位回归模型
    model_quantile.fit(X_train, y_train)  # 训练模型
    models[quantile] = model_quantile  # 存储每个分位数的模型

# 预测未来28天的销量
forecast_days = 28  # 预测天数
future_dates = pd.date_range(start=calendar['date'].max() + pd.Timedelta(days=1), periods=forecast_days)

# 准备未来数据并逐天进行预测
future_data = []
last_data = sales_train_melted[sales_train_melted['date'] == sales_train_melted['date'].max()].copy()

# 逐日进行销量预测
for i in range(forecast_days):
    current_date = future_dates[i]  # 获取当前日期
    current_calendar = calendar[calendar['date'] == current_date]  # 获取当前日期的日历数据
    
    # 如果当前日期的日历数据为空，使用默认值填充
    if current_calendar.empty:
        current_calendar = pd.DataFrame({
            'date': [current_date],
            'wm_yr_wk': [calendar['wm_yr_wk'].max()],
            'event_name_1': [None],
            'event_type_1': [None],
            'snap_CA': [0],
            'snap_TX': [0],
            'snap_WI': [0]
        })
    
    # 使用所有分位数模型对当前日期进行预测
    day_predictions = {}
    for quantile in quantiles:
        model = models[quantile]  # 获取当前分位数的模型
        X_future = last_data.drop(columns=['sales', 'date'])  # 删除目标列和日期列，准备预测数据
        day_predictions[quantile] = model.predict(X_future)  # 进行预测
    
    # 将预测结果存储
    future_data.append({
        'date': current_date,
        'pred_50': day_predictions[0.5],  # 50%分位数预测
        'pred_67': day_predictions[0.67],  # 67%分位数预测
        'pred_95': day_predictions[0.95],  # 95%分位数预测
        'pred_99': day_predictions[0.99]   # 99%分位数预测
    })

# 将未来预测结果转换为 DataFrame
forecast_df = pd.DataFrame(future_data)

# 按要求格式生成最终提交文件
def generate_submission_quantiles(predictions, sample_submission):
    new_cols = [f'F{i}' for i in range(1, 29)]  # 为28天的预测结果重命名列名
    prediction_data = []

    # 为每个分位数生成提交数据
    for quantile in quantiles:
        quantile_predictions = predictions.copy()  # 复制预测结果
        quantile_predictions = quantile_predictions.rename({k: v for k, v in zip(quantile_predictions.columns[1:], new_cols)}, axis=1)
        quantile_predictions['id'] = predictions['id'].apply(lambda x: f'{x}_quantile{int(quantile * 100)}')  # 为id添加分位数信息
        prediction_data.append(quantile_predictions)

    # 合并所有分位数的预测结果
    final_predictions = pd.concat(prediction_data, axis=0)

    # 将预测结果与sample_submission文件合并，符合提交格式要求
    submission = pd.merge(sample_submission, final_predictions, on='id', how='left')
    
    return submission

# 读取样例提交文件
sample_submission_path = '/mnt/data/sample_submission.csv'
sample_submission = pd.read_csv(sample_submission_path)

# 生成最终提交文件
final_submission = generate_submission_quantiles(forecast_df, sample_submission)

# 保存最终提交文件为CSV
submission_file_path = 'final_submission_uncertainty.csv'
final_submission.to_csv(submission_file_path, index=False)

print(f"Submission saved to {submission_file_path}")
