"""
================================================================================
项目一：基于多因子模型与机器学习的A股量化策略研究
技术栈：Pandas, NumPy, Backtrader, Scikit-learn, LightGBM, 因子分析, 机器学习
数据源：AKShare (免费A股数据接口)
================================================================================
"""
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools
from tqdm import tqdm
import time

plt.style.use('seaborn-v0_8-darkgrid')

import akshare as ak

# 机器学习库
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import lightgbm as lgb
import xgboost as xgb

# 回测框架
import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds

# 统计和优化
from scipy.stats import zscore, spearmanr, pearsonr, ttest_ind
from scipy.optimize import minimize, Bounds
from scipy import linalg
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


class AShareDataFetcher:
    """A股数据获取与处理模块"""

    def __init__(self, use_real_data=True, data_source='akshare'):
        self.use_real_data = use_real_data
        self.data_source = data_source
        self.stock_data = {}
        self.factor_data = {}
        self.returns_data = {}
        self.tickers = []
        self.market_data = None
        self.industry_data = {}

    def fetch_a_share_data(self, symbol_list=None, start_date='20190101', end_date='20231231'):
        """
        获取A股股票数据
        展示技能：API调用、数据清洗、时间序列处理
        """
        print("=" * 60)
        print("1. 获取A股市场数据")
        print("=" * 60)

        if not self.use_real_data or not AKSHARE_AVAILABLE:
            print("使用模拟数据...")
            return self._generate_simulated_data()

        if symbol_list is None:
            # 获取沪深300成分股
            print("获取沪深300成分股...")
            try:
                hs300 = ak.index_stock_cons_sina("000300")
                symbol_list = hs300['code'].tolist()[:50]  # 取前50只
                self.tickers = symbol_list
                print(f"获取到 {len(symbol_list)} 只沪深300成分股")
            except:
                print("无法获取沪深300成分股，使用默认股票列表")
                symbol_list = ['000001', '000002', '000858', '002415', '300750']
                self.tickers = symbol_list

        all_data = {}

        for symbol in tqdm(symbol_list, desc="下载A股数据"):
            try:
                if self.data_source == 'akshare':
                    # 使用AKShare获取数据
                    stock_zh_a_hist_df = ak.stock_zh_a_hist(
                        symbol=symbol,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"  # 前复权
                    )

                    if not stock_zh_a_hist_df.empty:
                        # 重命名列
                        stock_zh_a_hist_df = stock_zh_a_hist_df.rename(columns={
                            '日期': 'Date',
                            '开盘': 'Open',
                            '收盘': 'Close',
                            '最高': 'High',
                            '最低': 'Low',
                            '成交量': 'Volume',
                            '成交额': 'Amount',
                            '振幅': 'Amplitude',
                            '涨跌幅': 'Pct_Change',
                            '涨跌额': 'Change',
                            '换手率': 'Turnover'
                        })

                        # 设置日期索引
                        stock_zh_a_hist_df['Date'] = pd.to_datetime(stock_zh_a_hist_df['Date'])
                        stock_zh_a_hist_df.set_index('Date', inplace=True)

                        # 计算收益率
                        stock_zh_a_hist_df['Returns'] = stock_zh_a_hist_df['Close'].pct_change()
                        stock_zh_a_hist_df['Log_Returns'] = np.log(
                            stock_zh_a_hist_df['Close'] / stock_zh_a_hist_df['Close'].shift(1))

                        all_data[symbol] = stock_zh_a_hist_df

                elif self.data_source == 'tushare' and TUSHARE_AVAILABLE:
                    # 使用Tushare获取数据
                    ts.set_token('你的token')  # 需要替换为你的token
                    pro = ts.pro_api()

                    df = pro.daily(ts_code=f"{symbol}.SZ" if symbol.startswith('0') else f"{symbol}.SH",
                                   start_date=start_date, end_date=end_date)

                    if not df.empty:
                        df['trade_date'] = pd.to_datetime(df['trade_date'])
                        df.set_index('trade_date', inplace=True)
                        df.sort_index(inplace=True)

                        # 计算收益率
                        df['returns'] = df['close'].pct_change()
                        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

                        all_data[symbol] = df

            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
                continue

            # 避免请求过于频繁
            time.sleep(0.1)

        self.stock_data = all_data

        # 获取市场指数数据
        print("\n获取市场指数数据...")
        try:
            # 获取上证指数
            sh_index = ak.stock_zh_index_hist(symbol="000001", period="daily",
                                              start_date=start_date, end_date=end_date)
            sh_index['日期'] = pd.to_datetime(sh_index['日期'])
            sh_index.set_index('日期', inplace=True)
            self.market_data = sh_index
            print(f"获取上证指数数据: {len(sh_index)} 个交易日")
        except Exception as e:
            print(f"获取指数数据失败: {e}")

        print(f"\n数据获取完成:")
        print(f"成功获取 {len(all_data)} 只A股数据")
        print(f"数据时间范围: {start_date} 到 {end_date}")

        return self

    def _generate_simulated_data(self, n_stocks=50, n_days=1000):
        """生成模拟A股数据"""
        print("生成模拟A股数据...")

        np.random.seed(42)

        # 生成股票代码（模拟A股代码）
        self.tickers = [f'{i:06d}' for i in range(1, n_stocks + 1)]

        # 生成日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=n_days)
        dates = pd.date_range(start_date, end_date, freq='B')

        all_data = {}

        for ticker in self.tickers:
            # 生成价格序列
            initial_price = np.random.uniform(5, 100)
            drift = np.random.normal(0.0002, 0.0001)
            volatility = np.random.uniform(0.015, 0.035)

            # 几何布朗运动
            returns = np.random.normal(drift, volatility, len(dates))
            price_series = initial_price * np.exp(np.cumsum(returns))

            # 添加市场相关性
            market_factor = np.sin(np.linspace(0, 4 * np.pi, len(dates))) * np.random.uniform(2, 5)
            price_series = price_series + market_factor

            # 生成成交量
            volume_series = np.random.lognormal(14, 1, len(dates))

            # 创建DataFrame
            df = pd.DataFrame({
                'Open': price_series * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
                'High': price_series * (1 + np.random.uniform(0, 0.02, len(dates))),
                'Low': price_series * (1 - np.random.uniform(0, 0.02, len(dates))),
                'Close': price_series,
                'Volume': volume_series,
                'Amount': price_series * volume_series,
                'Pct_Change': np.random.normal(0, 0.02, len(dates)),
                'Turnover': np.random.uniform(0.5, 5, len(dates))
            }, index=dates)

            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))

            all_data[ticker] = df

        self.stock_data = all_data

        # 生成模拟市场指数
        market_prices = np.cumprod(1 + np.random.normal(0.0001, 0.015, len(dates)))
        self.market_data = pd.DataFrame({
            'close': market_prices * 3000,
            'returns': np.random.normal(0.0001, 0.015, len(dates))
        }, index=dates)

        return self

    def calculate_factors(self):
        """
        计算A股多因子
        展示技能：因子计算、特征工程、Pandas高级操作
        """
        print("\n" + "=" * 60)
        print("2. 计算A股多因子")
        print("=" * 60)

        if not self.stock_data:
            print("错误: 没有股票数据")
            return self

        all_factors = {}

        for ticker, data in tqdm(self.stock_data.items(), desc="计算因子"):
            try:
                factors = pd.DataFrame(index=data.index)
                close_prices = data['Close']
                returns = data['Returns'].dropna()
                volumes = data['Volume']

                # === 1. 动量因子 ===
                # 价格动量
                factors['MOM_1M'] = close_prices.pct_change(20)  # 1个月动量
                factors['MOM_3M'] = close_prices.pct_change(60)  # 3个月动量
                factors['MOM_6M'] = close_prices.pct_change(120)  # 6个月动量
                factors['MOM_12M'] = close_prices.pct_change(250)  # 12个月动量

                # 收益动量
                factors['RET_MOM_1M'] = returns.rolling(20).mean()  # 1个月收益动量
                factors['RET_MOM_3M'] = returns.rolling(60).mean()  # 3个月收益动量

                # 反转因子
                factors['REV_1W'] = -close_prices.pct_change(5)  # 1周反转
                factors['REV_1M'] = -close_prices.pct_change(20)  # 1个月反转

                # === 2. 波动率因子 ===
                factors['VOL_20D'] = returns.rolling(20).std() * np.sqrt(252)  # 20日年化波动率
                factors['VOL_60D'] = returns.rolling(60).std() * np.sqrt(252)  # 60日年化波动率
                factors['IDIO_VOL'] = self._calculate_idiosyncratic_volatility(data, self.market_data)

                # 最大回撤
                factors['MAX_DD_20D'] = self._calculate_max_drawdown(close_prices, window=20)
                factors['MAX_DD_60D'] = self._calculate_max_drawdown(close_prices, window=60)

                # 偏度和峰度
                factors['SKEW_20D'] = returns.rolling(20).skew()
                factors['KURT_20D'] = returns.rolling(20).kurt()

                # === 3. 流动性因子 ===
                # 换手率
                if 'Turnover' in data.columns:
                    factors['TURN_1M'] = data['Turnover'].rolling(20).mean()
                    factors['TURN_3M'] = data['Turnover'].rolling(60).mean()
                else:
                    # 计算近似换手率
                    turnover = volumes / volumes.rolling(20).mean()
                    factors['TURN_1M'] = turnover.rolling(20).mean()

                # Amihud非流动性指标
                factors['AMIHUD'] = self._calculate_amihud_illiquidity(data)

                # 成交量比率
                factors['VOL_RATIO'] = volumes / volumes.rolling(20).mean()

                # === 4. 技术指标因子 ===
                # RSI
                factors['RSI_14'] = self._calculate_rsi(close_prices, period=14)

                # MACD
                macd, signal, hist = self._calculate_macd(close_prices)
                factors['MACD'] = macd
                factors['MACD_SIGNAL'] = signal
                factors['MACD_HIST'] = hist

                # 布林带
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices, window=20)
                factors['BB_WIDTH'] = (bb_upper - bb_lower) / bb_middle
                factors['BB_POSITION'] = (close_prices - bb_lower) / (bb_upper - bb_lower)

                # 移动平均
                sma_20 = close_prices.rolling(20).mean()
                sma_60 = close_prices.rolling(60).mean()
                factors['SMA_RATIO_20_60'] = sma_20 / sma_60 - 1

                # === 5. 估值因子（模拟）===
                # 在真实项目中，这些需要从财务报表获取
                factors['PE_RATIO'] = np.random.lognormal(2.5, 0.5, len(factors))  # 模拟PE
                factors['PB_RATIO'] = np.random.lognormal(1, 0.3, len(factors))  # 模拟PB
                factors['PS_RATIO'] = np.random.lognormal(2, 0.4, len(factors))  # 模拟PS

                # === 6. 质量因子（模拟）===
                factors['ROE'] = np.random.normal(0.1, 0.05, len(factors))  # 模拟ROE
                factors['ROA'] = np.random.normal(0.05, 0.02, len(factors))  # 模拟ROA
                factors['GROWTH'] = np.random.normal(0.15, 0.1, len(factors))  # 模拟营收增长率

                # === 7. 规模因子 ===
                # 市值（用价格*模拟流通股本）
                market_cap = close_prices * np.random.lognormal(18, 1, len(close_prices))
                factors['MARKET_CAP'] = market_cap
                factors['LOG_MKT_CAP'] = np.log(market_cap)

                all_factors[ticker] = factors

            except Exception as e:
                print(f"计算 {ticker} 因子失败: {e}")
                continue

        self.factor_data = all_factors

        print(f"\n因子计算完成:")
        print(f"为 {len(all_factors)} 只股票计算了因子")
        print(f"因子数量: {len(all_factors[list(all_factors.keys())[0]].columns) if all_factors else 0}")

        return self

    def _calculate_max_drawdown(self, prices, window=252):
        """计算滚动最大回撤"""
        rolling_max = prices.rolling(window, min_periods=1).max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown.rolling(window, min_periods=1).min()

    def _calculate_rsi(self, prices, period=14):
        """计算相对强弱指数(RSI)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """计算布林带"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band

    def _calculate_amihud_illiquidity(self, data):
        """计算Amihud非流动性指标"""
        returns = data['Returns'].abs()
        dollar_volume = data['Close'] * data['Volume']
        illiquidity = returns / dollar_volume
        return illiquidity.rolling(20).mean()

    def _calculate_idiosyncratic_volatility(self, stock_data, market_data, window=60):
        """计算 idiosyncratic volatility (残差波动率)"""
        idiosyncratic_vol = pd.Series(index=stock_data.index, dtype=float)

        for i in range(window, len(stock_data)):
            if i >= window:
                # 获取窗口期数据
                stock_returns = stock_data['Returns'].iloc[i - window:i]
                market_returns = market_data['returns'].iloc[i - window:i] if 'returns' in market_data.columns else \
                market_data['close'].pct_change().iloc[i - window:i]

                # 对齐数据
                aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
                if len(aligned_data) > 10:
                    X = aligned_data.iloc[:, 1].values.reshape(-1, 1)
                    y = aligned_data.iloc[:, 0].values

                    # 计算市场模型回归
                    X_with_const = sm.add_constant(X)
                    model = sm.OLS(y, X_with_const)
                    results = model.fit()

                    # 残差波动率
                    residuals = results.resid
                    idiosyncratic_vol.iloc[i] = np.std(residuals) * np.sqrt(252)

        return idiosyncratic_vol

    def create_factor_panel(self):
        """
        创建因子面板数据
        展示技能：面板数据处理、数据重塑
        """
        print("\n" + "=" * 60)
        print("3. 创建因子面板数据")
        print("=" * 60)

        if not self.factor_data:
            print("错误: 没有因子数据")
            return None

        # 收集所有因子数据
        factor_panels = {}

        for ticker, factors in self.factor_data.items():
            for factor_name in factors.columns:
                if factor_name not in factor_panels:
                    factor_panels[factor_name] = pd.DataFrame()
                factor_panels[factor_name][ticker] = factors[factor_name]

        # 创建收益率面板
        returns_panel = pd.DataFrame()
        for ticker, data in self.stock_data.items():
            if ticker in self.factor_data:
                returns_panel[ticker] = data['Returns']

        # 对齐所有面板的时间索引
        common_index = returns_panel.index
        for factor_name in factor_panels:
            factor_panels[factor_name] = factor_panels[factor_name].reindex(common_index)

        returns_panel = returns_panel.reindex(common_index)

        self.factor_panels = factor_panels
        self.returns_panel = returns_panel

        print(f"面板数据创建完成:")
        print(f"时间维度: {len(common_index)} 个交易日")
        print(f"横截面维度: {returns_panel.shape[1]} 只股票")
        print(f"因子数量: {len(factor_panels)}")

        return factor_panels, returns_panel

    def analyze_factor_performance(self):
        """
        分析因子表现
        展示技能：因子分析、统计检验、可视化
        """
        print("\n" + "=" * 60)
        print("4. 因子表现分析")
        print("=" * 60)

        if not hasattr(self, 'factor_panels') or not hasattr(self, 'returns_panel'):
            print("错误: 没有面板数据")
            return self

        factor_performance = {}

        for factor_name, factor_panel in tqdm(self.factor_panels.items(), desc="分析因子表现"):
            try:
                # 对齐因子和收益率数据
                aligned_factor = factor_panel.reindex(self.returns_panel.index)
                aligned_returns = self.returns_panel.reindex(factor_panel.index)

                # 计算IC (信息系数)
                ic_series = []
                ic_pvalues = []

                for date in aligned_factor.index[1:]:  # 跳过第一天
                    if date in aligned_factor.index and date in aligned_returns.index:
                        # 获取当期因子值和下期收益率
                        current_factors = aligned_factor.loc[:date].iloc[-1]  # 最新因子值
                        next_returns = aligned_returns.shift(-1).loc[date]  # 下期收益率

                        # 对齐数据
                        common_stocks = current_factors.dropna().index.intersection(next_returns.dropna().index)
                        if len(common_stocks) > 10:
                            factor_values = current_factors[common_stocks]
                            return_values = next_returns[common_stocks]

                            # 计算Rank IC (Spearman相关系数)
                            ic, pvalue = spearmanr(factor_values, return_values)
                            ic_series.append(ic)
                            ic_pvalues.append(pvalue)

                if ic_series:
                    ic_mean = np.mean(ic_series)
                    ic_std = np.std(ic_series)
                    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
                    ic_hit_rate = np.mean(np.array(ic_series) > 0)

                    factor_performance[factor_name] = {
                        'IC_mean': ic_mean,
                        'IC_std': ic_std,
                        'IC_IR': ic_ir,
                        'IC_hit_rate': ic_hit_rate,
                        'num_obs': len(ic_series)
                    }

            except Exception as e:
                print(f"分析因子 {factor_name} 失败: {e}")
                continue

        # 转换为DataFrame
        perf_df = pd.DataFrame(factor_performance).T
        perf_df = perf_df.sort_values('IC_IR', ascending=False)

        self.factor_performance = perf_df

        print("\n因子表现排名 (按IC_IR):")
        print(perf_df.head(20).round(4))

        # 可视化因子表现
        self._visualize_factor_performance(perf_df)

        return perf_df

    def _visualize_factor_performance(self, perf_df):
        """可视化因子表现"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. IC_IR排名
        top_factors = perf_df.head(20)
        axes[0, 0].barh(range(len(top_factors)), top_factors['IC_IR'].values)
        axes[0, 0].set_yticks(range(len(top_factors)))
        axes[0, 0].set_yticklabels(top_factors.index)
        axes[0, 0].set_xlabel('IC Information Ratio')
        axes[0, 0].set_title('Top 20 Factors by IC IR')
        axes[0, 0].grid(True, alpha=0.3, axis='x')

        # 2. IC均值分布
        axes[0, 1].hist(perf_df['IC_mean'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('IC Mean')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of IC Means')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. IC命中率
        axes[1, 0].hist(perf_df['IC_hit_rate'].dropna(), bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Random=0.5')
        axes[1, 0].set_xlabel('IC Hit Rate')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of IC Hit Rates')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. IC均值 vs IC波动率
        axes[1, 1].scatter(perf_df['IC_std'], perf_df['IC_mean'], alpha=0.6, s=50)
        axes[1, 1].set_xlabel('IC Standard Deviation')
        axes[1, 1].set_ylabel('IC Mean')
        axes[1, 1].set_title('IC Mean vs IC Std Dev')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)

        # 添加因子数量标注
        axes[1, 1].text(0.05, 0.95, f'Total Factors: {len(perf_df)}',
                        transform=axes[1, 1].transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('factor_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig


class MachineLearningFactorModel:
    """机器学习因子模型"""

    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        self.models = {}
        self.predictions = {}
        self.feature_importance = {}

    def prepare_ml_data(self, test_size=0.2):
        """
        准备机器学习数据
        展示技能：特征工程、数据预处理、时间序列分割
        """
        print("\n" + "=" * 60)
        print("5. 准备机器学习数据")
        print("=" * 60)

        if not hasattr(self.data_fetcher, 'factor_panels'):
            print("错误: 没有因子面板数据")
            return None

        # 收集所有特征
        all_features = []
        feature_names = []

        for factor_name, factor_panel in self.data_fetcher.factor_panels.items():
            all_features.append(factor_panel.values.flatten(order='F'))
            feature_names.append(factor_name)

        # 创建特征矩阵 (时间 * 股票, 特征)
        X = np.column_stack(all_features)

        # 创建目标变量 (下一期收益率)
        returns_flat = self.data_fetcher.returns_panel.values.flatten(order='F')

        # 对齐数据
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(returns_flat)
        X = X[valid_mask]
        y = returns_flat[valid_mask]

        # 创建时间索引
        dates = self.data_fetcher.returns_panel.index
        stocks = self.data_fetcher.returns_panel.columns

        # 时间序列分割
        n_samples = len(X)
        split_idx = int(n_samples * (1 - test_size))

        # 打乱前保存原始索引
        indices = np.arange(n_samples)

        # 时间序列分割（不打乱）
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 特征缩放
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names

        print(f"数据准备完成:")
        print(f"总样本数: {n_samples}")
        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        print(f"特征数量: {len(feature_names)}")
        print(f"特征: {', '.join(feature_names[:10])}...")

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

    def train_models(self):
        """
        训练多种机器学习模型
        展示技能：机器学习建模、超参数调优、集成学习
        """
        print("\n" + "=" * 60)
        print("6. 训练机器学习模型")
        print("=" * 60)

        models = {}

        # 1. 线性回归
        print("\n1. 训练线性回归模型...")
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        models['LinearRegression'] = lr

        # 2. 岭回归
        print("2. 训练岭回归模型...")
        ridge = Ridge(alpha=1.0)
        ridge.fit(self.X_train, self.y_train)
        models['Ridge'] = ridge

        # 3. Lasso回归
        print("3. 训练Lasso回归模型...")
        lasso = Lasso(alpha=0.01)
        lasso.fit(self.X_train, self.y_train)
        models['Lasso'] = lasso

        # 4. 随机森林
        print("4. 训练随机森林模型...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        models['RandomForest'] = rf

        # 5. Gradient Boosting
        print("5. 训练Gradient Boosting模型...")
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        gb.fit(self.X_train, self.y_train)
        models['GradientBoosting'] = gb

        # 6. LightGBM
        print("6. 训练LightGBM模型...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        lgb_model.fit(self.X_train, self.y_train)
        models['LightGBM'] = lgb_model

        # 7. XGBoost
        print("7. 训练XGBoost模型...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(self.X_train, self.y_train)
        models['XGBoost'] = xgb_model

        self.models = models

        # 评估模型
        self.evaluate_models()

        return models

    def evaluate_models(self):
        """评估模型性能"""
        print("\n" + "=" * 60)
        print("7. 模型性能评估")
        print("=" * 60)

        results = []

        for name, model in self.models.items():
            # 训练集预测
            y_train_pred = model.predict(self.X_train)
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)

            # 测试集预测
            y_test_pred = model.predict(self.X_test)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)

            # 计算IC
            ic_test, _ = pearsonr(y_test_pred, self.y_test)

            results.append({
                'Model': name,
                'Train_MSE': train_mse,
                'Train_R2': train_r2,
                'Test_MSE': test_mse,
                'Test_R2': test_r2,
                'Test_IC': ic_test
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test_R2', ascending=False)

        print("\n模型性能比较:")
        print(results_df.round(4).to_string(index=False))

        # 可视化结果
        self._visualize_model_performance(results_df)

        # 特征重要性分析
        self.analyze_feature_importance()

        return results_df

    def _visualize_model_performance(self, results_df):
        """可视化模型性能"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. 测试集R2比较
        axes[0].barh(range(len(results_df)), results_df['Test_R2'].values)
        axes[0].set_yticks(range(len(results_df)))
        axes[0].set_yticklabels(results_df['Model'])
        axes[0].set_xlabel('R² Score (Test)')
        axes[0].set_title('Model Performance (R²)')
        axes[0].grid(True, alpha=0.3, axis='x')

        # 2. 训练集vs测试集R2
        x = range(len(results_df))
        width = 0.35
        axes[1].bar([i - width / 2 for i in x], results_df['Train_R2'], width, label='Train', alpha=0.8)
        axes[1].bar([i + width / 2 for i in x], results_df['Test_R2'], width, label='Test', alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
        axes[1].set_ylabel('R² Score')
        axes[1].set_title('Train vs Test R²')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        # 3. IC比较
        axes[2].barh(range(len(results_df)), results_df['Test_IC'].values)
        axes[2].set_yticks(range(len(results_df)))
        axes[2].set_yticklabels(results_df['Model'])
        axes[2].set_xlabel('Information Coefficient (Test)')
        axes[2].set_title('Model IC Performance')
        axes[2].grid(True, alpha=0.3, axis='x')
        axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n分析特征重要性...")

        # 获取最佳模型
        best_model_name = list(self.models.keys())[0]
        best_model = self.models[best_model_name]

        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            importances = np.abs(best_model.coef_)
        else:
            print(f"模型 {best_model_name} 没有特征重要性属性")
            return

        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

        self.feature_importance = feature_importance_df

        print(f"\n{best_model_name} 模型特征重要性 Top 20:")
        print(feature_importance_df.head(20).round(4).to_string(index=False))

        # 可视化特征重要性
        self._visualize_feature_importance(feature_importance_df, best_model_name)

        return feature_importance_df

    def _visualize_feature_importance(self, feature_importance_df, model_name):
        """可视化特征重要性"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 1. Top 20特征重要性
        top_20 = feature_importance_df.head(20)
        axes[0].barh(range(len(top_20)), top_20['Importance'].values)
        axes[0].set_yticks(range(len(top_20)))
        axes[0].set_yticklabels(top_20['Feature'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title(f'{model_name} - Top 20 Feature Importance')
        axes[0].grid(True, alpha=0.3, axis='x')

        # 2. 特征重要性累积分布
        cumulative_importance = np.cumsum(feature_importance_df['Importance'])
        cumulative_importance = cumulative_importance / cumulative_importance.iloc[-1]

        axes[1].plot(range(1, len(cumulative_importance) + 1), cumulative_importance.values,
                     linewidth=2)
        axes[1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Importance')
        axes[1].axvline(x=np.where(cumulative_importance >= 0.8)[0][0] + 1,
                        color='green', linestyle='--', alpha=0.7,
                        label=f'Top {np.where(cumulative_importance >= 0.8)[0][0] + 1} features')
        axes[1].set_xlabel('Number of Features')
        axes[1].set_ylabel('Cumulative Importance')
        axes[1].set_title('Cumulative Feature Importance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_ml_factors(self):
        """
        创建机器学习生成的因子
        展示技能：因子合成、特征组合
        """
        print("\n" + "=" * 60)
        print("8. 创建机器学习因子")
        print("=" * 60)

        ml_factors = {}

        for model_name, model in self.models.items():
            print(f"创建 {model_name} 因子...")

            # 对每个时间点预测
            predictions_by_date = {}

            for date in tqdm(self.data_fetcher.returns_panel.index, desc=f"预测 {model_name}", leave=False):
                # 获取当天的所有特征
                date_features = []
                for factor_name in self.feature_names:
                    factor_panel = self.data_fetcher.factor_panels[factor_name]
                    if date in factor_panel.index:
                        date_features.append(factor_panel.loc[date].values)

                if date_features:
                    X_date = np.column_stack(date_features).T

                    # 处理缺失值
                    valid_mask = ~np.isnan(X_date).any(axis=1)
                    X_date_valid = X_date[valid_mask]

                    if len(X_date_valid) > 0:
                        # 缩放特征
                        X_date_scaled = self.scaler.transform(X_date_valid)

                        # 预测
                        predictions = model.predict(X_date_scaled)

                        # 创建预测Series
                        pred_series = pd.Series(index=self.data_fetcher.returns_panel.columns, dtype=float)
                        pred_series.iloc[valid_mask] = predictions

                        predictions_by_date[date] = pred_series

            # 转换为DataFrame
            if predictions_by_date:
                ml_factor = pd.DataFrame(predictions_by_date).T
                ml_factors[f'ML_{model_name}'] = ml_factor

        self.ml_factors = ml_factors

        print(f"\n机器学习因子创建完成:")
        print(f"生成了 {len(ml_factors)} 个机器学习因子")

        return ml_factors


class BacktraderStrategy(bt.Strategy):
    """
    Backtrader多因子策略
    展示技能：回测框架、交易逻辑、风险控制
    """

    params = (
        ('top_n', 20),  # 买入前N只
        ('bottom_n', 20),  # 卖空前N只
        ('rebalance_days', 20),  # 调仓周期
        ('max_position_pct', 0.05),  # 单只股票最大仓位
        ('stop_loss_pct', 0.10),  # 止损比例
        ('take_profit_pct', 0.20),  # 止盈比例
    )

    def __init__(self, factor_data=None, ml_predictions=None):
        super().__init__()

        self.factor_data = factor_data
        self.ml_predictions = ml_predictions
        self.day_counter = 0
        self.positions_info = {}  # 跟踪持仓信息

        # 添加技术指标
        self.sma = {}
        for data in self.datas:
            self.sma[data._name] = bt.indicators.SimpleMovingAverage(data, period=20)

    def next(self):
        self.day_counter += 1

        # 检查止损/止盈
        self.check_stop_loss_take_profit()

        # 调仓日
        if self.day_counter % self.params.rebalance_days == 0:
            self.rebalance_portfolio()

    def rebalance_portfolio(self):
        """调仓逻辑"""
        current_date = self.datas[0].datetime.date(0)

        if self.factor_data is None:
            # 如果没有外部因子数据，使用价格动量
            stock_scores = self.calculate_price_momentum()
        else:
            # 使用因子数据
            stock_scores = self.calculate_factor_scores(current_date)

        if stock_scores is None or len(stock_scores) == 0:
            return

        # 按分数排序
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)

        # 买入信号（分数最高的）
        long_stocks = [s[0] for s in sorted_stocks[:self.params.top_n]]

        # 卖空信号（分数最低的）
        short_stocks = [s[0] for s in sorted_stocks[-self.params.bottom_n:]]

        # 获取当前持仓
        current_positions = {}
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size != 0:
                current_positions[data._name] = pos.size

        # 平掉不在新组合中的头寸
        for stock_name, pos_size in list(current_positions.items()):
            if stock_name not in long_stocks and stock_name not in short_stocks:
                data = [d for d in self.datas if d._name == stock_name][0]
                self.close(data)
                if stock_name in self.positions_info:
                    del self.positions_info[stock_name]

        # 计算每只股票的权重
        n_long = len(long_stocks)
        n_short = len(short_stocks)

        if n_long > 0:
            long_weight = 0.7 / n_long  # 70%资金用于做多
        else:
            long_weight = 0

        if n_short > 0:
            short_weight = -0.3 / n_short  # 30%资金用于做空
        else:
            short_weight = 0

        # 建立新的多头头寸
        for stock_name in long_stocks:
            data = [d for d in self.datas if d._name == stock_name][0]
            current_pos = self.getposition(data).size

            if current_pos == 0:  # 开新仓
                target_size = (self.broker.getvalue() * long_weight) / data.close[0]
                target_shares = int(target_size / 100) * 100  # A股以手为单位

                if target_shares > 0:
                    self.buy(data=data, size=target_shares)
                    # 记录入场信息
                    self.positions_info[stock_name] = {
                        'entry_price': data.close[0],
                        'entry_date': current_date,
                        'position_type': 'long'
                    }
            elif current_pos > 0:  # 已有持仓，调整到目标仓位
                target_size = (self.broker.getvalue() * long_weight) / data.close[0]
                target_shares = int(target_size / 100) * 100

                if target_shares > current_pos:  # 加仓
                    add_shares = target_shares - current_pos
                    self.buy(data=data, size=add_shares)
                elif target_shares < current_pos:  # 减仓
                    reduce_shares = current_pos - target_shares
                    self.sell(data=data, size=reduce_shares)

        # 建立新的空头头寸
        for stock_name in short_stocks:
            data = [d for d in self.datas if d._name == stock_name][0]
            current_pos = self.getposition(data).size

            if current_pos == 0:  # 开新仓
                target_size = (self.broker.getvalue() * abs(short_weight)) / data.close[0]
                target_shares = int(target_size / 100) * 100

                if target_shares > 0:
                    self.sell(data=data, size=target_shares)
                    # 记录入场信息
                    self.positions_info[stock_name] = {
                        'entry_price': data.close[0],
                        'entry_date': current_date,
                        'position_type': 'short'
                    }
            elif current_pos < 0:  # 已有空头持仓，调整到目标仓位
                target_size = (self.broker.getvalue() * abs(short_weight)) / data.close[0]
                target_shares = int(target_size / 100) * 100
                current_shares_abs = abs(current_pos)

                if target_shares > current_shares_abs:  # 加空仓
                    add_shares = target_shares - current_shares_abs
                    self.sell(data=data, size=add_shares)
                elif target_shares < current_shares_abs:  # 减空仓
                    reduce_shares = current_shares_abs - target_shares
                    self.buy(data=data, size=reduce_shares)

    def calculate_price_momentum(self):
        """计算价格动量分数（备用方法）"""
        scores = {}

        for data in self.datas:
            if len(data) > 20:
                # 计算20日收益率作为动量
                momentum = (data.close[0] - data.close[-20]) / data.close[-20]
                scores[data._name] = momentum

        return scores

    def calculate_factor_scores(self, current_date):
        """计算因子分数"""
        scores = {}

        # 将datetime.date转换为字符串
        date_str = current_date.strftime('%Y-%m-%d')

        for data in self.datas:
            if data._name in self.ml_predictions.columns:
                # 查找最近的预测日期
                available_dates = [d for d in self.ml_predictions.index if d <= current_date]
                if available_dates:
                    nearest_date = max(available_dates)
                    if nearest_date in self.ml_predictions.index:
                        score = self.ml_predictions.loc[nearest_date, data._name]
                        if not np.isnan(score):
                            scores[data._name] = score

        return scores

    def check_stop_loss_take_profit(self):
        """检查止损和止盈"""
        current_date = self.datas[0].datetime.date(0)

        for stock_name, info in list(self.positions_info.items()):
            data = [d for d in self.datas if d._name == stock_name][0]
            current_price = data.close[0]
            entry_price = info['entry_price']

            if info['position_type'] == 'long':
                # 多头持仓
                returns_pct = (current_price - entry_price) / entry_price

                if returns_pct <= -self.params.stop_loss_pct:
                    # 止损
                    self.close(data=data)
                    print(f"{current_date}: {stock_name} 触发止损，收益率: {returns_pct:.2%}")
                    del self.positions_info[stock_name]
                elif returns_pct >= self.params.take_profit_pct:
                    # 止盈
                    self.close(data=data)
                    print(f"{current_date}: {stock_name} 触发止盈，收益率: {returns_pct:.2%}")
                    del self.positions_info[stock_name]

            elif info['position_type'] == 'short':
                # 空头持仓
                returns_pct = (entry_price - current_price) / entry_price

                if returns_pct <= -self.params.stop_loss_pct:
                    # 止损
                    self.close(data=data)
                    print(f"{current_date}: {stock_name} 空头触发止损，收益率: {returns_pct:.2%}")
                    del self.positions_info[stock_name]
                elif returns_pct >= self.params.take_profit_pct:
                    # 止盈
                    self.close(data=data)
                    print(f"{current_date}: {stock_name} 空头触发止盈，收益率: {returns_pct:.2%}")
                    del self.positions_info[stock_name]

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.data._name}, '
                         f'Price: {order.executed.price:.2f}, '
                         f'Cost: {order.executed.value:.2f}, '
                         f'Comm: {order.executed.comm:.2f}')
            else:
                self.log(f'SELL EXECUTED, {order.data._name}, '
                         f'Price: {order.executed.price:.2f}, '
                         f'Cost: {order.executed.value:.2f}, '
                         f'Comm: {order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.data._name}')

    def log(self, txt, dt=None):
        """日志记录"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')


def run_backtest(data_fetcher, ml_model, initial_cash=1000000):
    """
    运行回测
    展示技能：回测系统搭建、绩效分析
    """
    print("\n" + "=" * 60)
    print("9. 运行Backtrader回测")
    print("=" * 60)

    # 创建Cerebro引擎
    cerebro = bt.Cerebro()

    # 设置初始资金
    cerebro.broker.setcash(initial_cash)

    # 设置交易成本
    cerebro.broker.setcommission(commission=0.001)  # 0.1%手续费
    cerebro.broker.set_slippage_perc(0.001)  # 0.1%滑点

    # 添加策略
    if hasattr(ml_model, 'ml_factors') and ml_model.ml_factors:
        # 使用机器学习因子
        best_ml_factor = list(ml_model.ml_factors.keys())[0]
        ml_predictions = ml_model.ml_factors[best_ml_factor]

        cerebro.addstrategy(
            BacktraderStrategy,
            factor_data=data_fetcher.factor_data,
            ml_predictions=ml_predictions,
            top_n=20,
            bottom_n=20,
            rebalance_days=20,
            max_position_pct=0.05,
            stop_loss_pct=0.10,
            take_profit_pct=0.20
        )
    else:
        # 使用传统因子
        cerebro.addstrategy(
            BacktraderStrategy,
            factor_data=data_fetcher.factor_data,
            ml_predictions=None
        )

    # 添加股票数据
    print("添加股票数据到回测引擎...")
    for ticker, data in tqdm(list(data_fetcher.stock_data.items())[:50], desc="加载数据"):  # 限制50只股票
        try:
            # 准备数据
            df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df.index)

            # 创建数据源
            bt_data = btfeeds.PandasData(
                dataname=df,
                name=ticker
            )

            cerebro.adddata(bt_data)

        except Exception as e:
            print(f"添加 {ticker} 数据失败: {e}")
            continue

    # 添加分析器
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02, annualize=True)
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(btanalyzers.VWR, _name='vwr')
    cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')

    # 运行回测
    print(f"初始资金: {cerebro.broker.getvalue():.2f}")

    results = cerebro.run()
    strat = results[0]

    # 最终资金
    final_value = cerebro.broker.getvalue()
    print(f"最终资金: {final_value:.2f}")
    print(f"总收益率: {(final_value / initial_cash - 1) * 100:.2f}%")

    # 获取分析结果
    returns_analyzer = strat.analyzers.returns.get_analysis()
    sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
    drawdown_analyzer = strat.analyzers.drawdown.get_analysis()
    trades_analyzer = strat.analyzers.trades.get_analysis()

    # 打印绩效指标
    print("\n" + "=" * 60)
    print("回测绩效指标")
    print("=" * 60)

    if 'rnorm100' in returns_analyzer:
        print(f"年化收益率: {returns_analyzer['rnorm100']:.2f}%")
    if 'sharperatio' in sharpe_analyzer:
        print(f"夏普比率: {sharpe_analyzer['sharperatio']:.3f}")
    if 'max' in drawdown_analyzer:
        print(f"最大回撤: {drawdown_analyzer['max']['drawdown']:.2f}%")
        print(f"最长回撤天数: {drawdown_analyzer['max']['len']}")

    if trades_analyzer:
        total_trades = trades_analyzer.total.total
        if total_trades > 0:
            print(f"总交易次数: {total_trades}")
            print(f"盈利交易比例: {trades_analyzer.won.total / total_trades:.2%}")

    # 绘制回测结果
    print("\n生成回测图表...")
    cerebro.plot(style='candle', barup='green', bardown='red',
                 volume=False, savefig=True,
                 figfilename='backtest_results.png')

    return cerebro, strat


def main():
    """主函数"""
    print("=" * 60)
    print("A股多因子量化策略研究项目")
    print("=" * 60)

    # 1. 数据获取
    data_fetcher = AShareDataFetcher(use_real_data=True, data_source='akshare')

    # 获取A股数据（使用部分沪深300成分股）
    symbol_list = ['000001', '000002', '000858', '002415', '300750',
                   '600519', '000333', '002475', '300059', '000651']

    data_fetcher.fetch_a_share_data(
        symbol_list=symbol_list,
        start_date='20200101',
        end_date='20231231'
    )

    # 2. 因子计算
    data_fetcher.calculate_factors()

    # 3. 创建因子面板
    data_fetcher.create_factor_panel()

    # 4. 因子分析
    factor_performance = data_fetcher.analyze_factor_performance()

    # 5. 机器学习模型
    ml_model = MachineLearningFactorModel(data_fetcher)

    # 准备机器学习数据
    ml_model.prepare_ml_data(test_size=0.3)

    # 训练模型
    ml_model.train_models()

    # 创建机器学习因子
    ml_factors = ml_model.create_ml_factors()

    # 6. 运行回测
    cerebro, strat = run_backtest(data_fetcher, ml_model, initial_cash=1000000)

    print("\n" + "=" * 60)
    print("项目总结:")
    print("1. 使用AKShare获取真实A股数据")
    print("2. 计算了动量、波动率、流动性、技术指标等多类因子")
    print("3. 进行了因子IC分析，筛选有效因子")
    print("4. 实现了线性回归、树模型、集成学习等多种机器学习模型")
    print("5. 使用Backtrader进行了完整的策略回测")
    print("6. 包含了止损止盈、仓位控制等风险管理机制")
    print("=" * 60)

    return data_fetcher, ml_model, cerebro


if __name__ == "__main__":
    # 安装所需库
    print(
        "安装所需库: pip install akshare tushare pandas numpy scipy scikit-learn lightgbm xgboost backtrader matplotlib seaborn tqdm statsmodels")

    data_fetcher, ml_model, cerebro = main()