"""
================================================================================
项目名称：投资组合风险计量与蒙特卡洛模拟
技术栈：Python, Pandas, NumPy, SciPy, Statsmodels, Backtrader, 蒙特卡洛模拟, VaR/CVaR/ES计算, 压力测试, 风险模型
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
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

# 回测框架
import backtrader as bt
import backtrader.analyzers as btanalyzers
import backtrader.feeds as btfeeds

# 数据获取
try:
    import akshare as ak
    import yfinance as yf

    DATA_AVAILABLE = True
    print("✓ 数据接口可用")
except ImportError:
    DATA_AVAILABLE = False
    print("⚠ 数据接口不可用，将使用模拟数据")


class PortfolioRiskManager:
    """投资组合风险管理与蒙特卡洛模拟系统"""

    def __init__(self, symbols=None, portfolio_weights=None):
        self.symbols = symbols or ['000300.SH', '000905.SH', '000016.SH']  # 沪深300, 中证500, 上证50
        self.portfolio_weights = portfolio_weights or np.ones(len(symbols)) / len(symbols) if symbols else None
        self.data = None
        self.returns = None
        self.log_returns = None
        self.cov_matrix = None
        self.corr_matrix = None
        self.portfolio_returns = None
        self.risk_metrics = {}

    def fetch_market_data(self, start_date='2018-01-01', end_date='2023-12-31',
                          use_real_data=True, data_source='akshare'):
        """
        获取市场数据
        支持A股、美股、ETF、商品等多资产类别
        """
        print("=" * 60)
        print("1. 获取市场数据")
        print("=" * 60)

        if not use_real_data or not DATA_AVAILABLE:
            print("使用模拟数据...")
            return self._generate_simulated_data()

        all_data = {}

        for symbol in tqdm(self.symbols, desc="下载市场数据"):
            try:
                if data_source == 'akshare' and symbol.endswith('.SH'):
                    # A股指数
                    code = symbol.replace('.SH', '')
                    df = ak.index_zh_a_hist(symbol=code, period="daily",
                                            start_date=start_date.replace('-', ''),
                                            end_date=end_date.replace('-', ''))
                    df.index = pd.to_datetime(df['日期'])
                    df = df.rename(columns={'开盘': 'Open', '收盘': 'Close',
                                            '最高': 'High', '最低': 'Low', '成交量': 'Volume'})
                elif data_source == 'yfinance':
                    # 国际资产
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(start=start_date, end=end_date)
                else:
                    continue

                if not df.empty:
                    all_data[symbol] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    time.sleep(0.5)  # 避免请求过于频繁

            except Exception as e:
                print(f"获取 {symbol} 数据失败: {e}")
                continue

        if all_data:
            self.data = pd.concat(all_data, axis=1, keys=all_data.keys())
            self._calculate_returns()
            print(f"数据获取完成: {len(self.returns)} 个交易日")
            print(f"资产数量: {len(self.symbols)}")
        else:
            print("无法获取真实数据，使用模拟数据")
            return self._generate_simulated_data()

        return self

    def _generate_simulated_data(self, n_assets=5, n_days=1000):
        """生成模拟市场数据"""
        print("生成模拟市场数据...")

        np.random.seed(42)
        dates = pd.date_range('2018-01-01', periods=n_days, freq='B')

        # 生成相关性的收益率
        n_assets = len(self.symbols) if self.symbols else n_assets

        # 定义资产类别和特性
        asset_classes = ['Equity', 'Bond', 'Commodity', 'RealEstate', 'Cash']
        if n_assets > len(asset_classes):
            asset_classes = asset_classes + ['Equity'] * (n_assets - len(asset_classes))

        # 生成相关矩阵
        base_corr = 0.3
        corr_matrix = np.full((n_assets, n_assets), base_corr)
        np.fill_diagonal(corr_matrix, 1)

        # 相同资产类别相关性更高
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if asset_classes[i] == asset_classes[j]:
                    corr_matrix[i, j] = 0.7
                    corr_matrix[j, i] = 0.7

        # 生成收益率
        means = []
        volatilities = []

        for asset_class in asset_classes:
            if asset_class == 'Equity':
                means.append(0.0003)  # 年化约7.5%
                volatilities.append(0.02)  # 年化约32%
            elif asset_class == 'Bond':
                means.append(0.0001)  # 年化约2.5%
                volatilities.append(0.005)  # 年化约8%
            elif asset_class == 'Commodity':
                means.append(0.0002)  # 年化约5%
                volatilities.append(0.015)  # 年化约24%
            elif asset_class == 'RealEstate':
                means.append(0.00025)  # 年化约6%
                volatilities.append(0.01)  # 年化约16%
            else:  # Cash
                means.append(0.00004)  # 年化约1%
                volatilities.append(0.001)  # 年化约1.6%

        means = np.array(means)
        volatilities = np.array(volatilities)

        # 从相关矩阵生成协方差矩阵
        std_matrix = np.diag(volatilities)
        cov_matrix = std_matrix @ corr_matrix @ std_matrix

        # 生成多变量正态收益率
        returns = np.random.multivariate_normal(means, cov_matrix, n_days)

        # 生成价格序列
        initial_prices = np.random.uniform(50, 200, n_assets)
        price_series = initial_prices * np.exp(np.cumsum(returns, axis=0))

        # 添加跳跃和波动率聚集
        for i in range(n_assets):
            # 添加随机跳跃
            jump_days = np.random.choice(n_days, size=int(n_days * 0.01), replace=False)
            price_series[jump_days, i] *= np.random.uniform(0.9, 1.1, len(jump_days))

            # 添加GARCH效应
            garch_vol = volatilities[i] * np.ones(n_days)
            for t in range(1, n_days):
                garch_vol[t] = 0.1 + 0.8 * garch_vol[t - 1] ** 2 + 0.1 * returns[t - 1, i] ** 2
            price_series[:, i] *= np.exp(np.random.normal(0, garch_vol))

        # 创建DataFrame
        all_data = {}
        for i in range(n_assets):
            symbol = self.symbols[i] if i < len(self.symbols) else f'Asset_{i + 1}'
            df = pd.DataFrame({
                'Open': price_series[:, i] * np.random.uniform(0.99, 1.01, n_days),
                'High': price_series[:, i] * np.random.uniform(1.0, 1.02, n_days),
                'Low': price_series[:, i] * np.random.uniform(0.98, 1.0, n_days),
                'Close': price_series[:, i],
                'Volume': np.random.lognormal(14, 1, n_days)
            }, index=dates)
            all_data[symbol] = df

        self.data = pd.concat(all_data, axis=1, keys=all_data.keys())
        self.symbols = list(all_data.keys())
        self.portfolio_weights = np.ones(n_assets) / n_assets

        self._calculate_returns()

        print(f"\n模拟数据生成完成:")
        print(f"资产数量: {n_assets}")
        print(f"交易日数: {n_days}")
        print(f"时间范围: {dates[0].date()} 到 {dates[-1].date()}")

        return self

    def _calculate_returns(self):
        """计算收益率"""
        if self.data is None:
            return

        # 提取收盘价
        close_prices = self.data.xs('Close', axis=1, level=1)

        # 计算简单收益率和对数收益率
        self.returns = close_prices.pct_change().dropna()
        self.log_returns = np.log(close_prices / close_prices.shift(1)).dropna()

        # 计算协方差和相关系数矩阵
        self.cov_matrix = self.returns.cov() * 252  # 年化
        self.corr_matrix = self.returns.corr()

        # 计算投资组合收益率
        if self.portfolio_weights is not None:
            self.portfolio_returns = (self.returns * self.portfolio_weights).sum(axis=1)
            self.portfolio_log_returns = (self.log_returns * self.portfolio_weights).sum(axis=1)

        # 计算基本统计
        self._calculate_basic_statistics()

    def _calculate_basic_statistics(self):
        """计算基本统计量"""
        if self.returns is None:
            return

        print("\n基本统计信息:")
        print("-" * 40)

        # 单个资产统计
        stats_df = pd.DataFrame(index=self.returns.columns)
        stats_df['年化收益率'] = self.returns.mean() * 252
        stats_df['年化波动率'] = self.returns.std() * np.sqrt(252)
        stats_df['夏普比率'] = (stats_df['年化收益率'] - 0.02) / stats_df['年化波动率']
        stats_df['偏度'] = self.returns.skew()
        stats_df['超额峰度'] = self.returns.kurtosis()
        stats_df['最大回撤'] = self.returns.apply(lambda x: self._calculate_max_drawdown(x))
        stats_df['VaR_95'] = self.returns.apply(lambda x: self._calculate_historical_var(x, 0.95))
        stats_df['CVaR_95'] = self.returns.apply(lambda x: self._calculate_historical_cvar(x, 0.95))

        print("单个资产统计:")
        print(stats_df.round(4).to_string())

        # 投资组合统计
        if self.portfolio_returns is not None:
            portfolio_stats = {
                '年化收益率': self.portfolio_returns.mean() * 252,
                '年化波动率': self.portfolio_returns.std() * np.sqrt(252),
                '夏普比率': (self.portfolio_returns.mean() * 252 - 0.02) / (
                            self.portfolio_returns.std() * np.sqrt(252)),
                '偏度': self.portfolio_returns.skew(),
                '超额峰度': self.portfolio_returns.kurtosis(),
                '最大回撤': self._calculate_max_drawdown(self.portfolio_returns),
                'VaR_95': self._calculate_historical_var(self.portfolio_returns, 0.95),
                'CVaR_95': self._calculate_historical_cvar(self.portfolio_returns, 0.95)
            }

            print("\n投资组合统计:")
            for key, value in portfolio_stats.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4%}" if key in ['年化收益率', '年化波动率', '最大回撤', 'VaR_95',
                                                           'CVaR_95'] else f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")

    def _calculate_max_drawdown(self, returns):
        """计算最大回撤"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_historical_var(self, returns, alpha=0.95):
        """计算历史VaR"""
        return np.percentile(returns, (1 - alpha) * 100)

    def _calculate_historical_cvar(self, returns, alpha=0.95):
        """计算历史CVaR"""
        var = self._calculate_historical_var(returns, alpha)
        return returns[returns <= var].mean()

    def calculate_risk_metrics(self, alpha=0.95, horizon=1, methods=['historical', 'parametric', 'monte_carlo']):
        """
        计算投资组合风险指标
        包含VaR, CVaR, 预期损失(ES)
        """
        print("\n" + "=" * 60)
        print("2. 计算投资组合风险指标")
        print("=" * 60)

        if self.portfolio_returns is None:
            print("错误: 没有投资组合数据")
            return

        returns = self.portfolio_returns
        log_returns = self.portfolio_log_returns

        results = {}

        # 1. 历史模拟法
        if 'historical' in methods:
            print("\n历史模拟法:")
            var_hist = self._calculate_historical_var(returns, alpha)
            cvar_hist = self._calculate_historical_cvar(returns, alpha)

            # 多期VaR
            var_hist_horizon = self._calculate_var_horizon(returns, alpha, horizon, method='historical')
            cvar_hist_horizon = self._calculate_cvar_horizon(returns, alpha, horizon, method='historical')

            results['historical'] = {
                'VaR': var_hist,
                'CVaR': cvar_hist,
                f'VaR_{horizon}d': var_hist_horizon,
                f'CVaR_{horizon}d': cvar_hist_horizon
            }

            print(f"  1天 {alpha * 100}% VaR: {var_hist:.4%}")
            print(f"  1天 {alpha * 100}% CVaR: {cvar_hist:.4%}")
            print(f"  {horizon}天 {alpha * 100}% VaR: {var_hist_horizon:.4%}")
            print(f"  {horizon}天 {alpha * 100}% CVaR: {cvar_hist_horizon:.4%}")

        # 2. 参数法
        if 'parametric' in methods:
            print("\n参数法 (正态分布):")
            var_param_norm = self._calculate_parametric_var(returns, alpha, distribution='normal')
            cvar_param_norm = self._calculate_parametric_cvar(returns, alpha, distribution='normal')

            print("\n参数法 (t分布):")
            var_param_t = self._calculate_parametric_var(returns, alpha, distribution='t')
            cvar_param_t = self._calculate_parametric_cvar(returns, alpha, distribution='t')

            results['parametric'] = {
                'normal': {'VaR': var_param_norm, 'CVaR': cvar_param_norm},
                't': {'VaR': var_param_t, 'CVaR': cvar_param_t}
            }

            print(f"  正态分布 VaR: {var_param_norm:.4%}")
            print(f"  正态分布 CVaR: {cvar_param_norm:.4%}")
            print(f"  t分布 VaR: {var_param_t:.4%}")
            print(f"  t分布 CVaR: {cvar_param_t:.4%}")

        # 3. 蒙特卡洛模拟
        if 'monte_carlo' in methods:
            print("\n蒙特卡洛模拟:")
            mc_results = self._monte_carlo_var(alpha=alpha, horizon=horizon, n_simulations=10000)
            results['monte_carlo'] = mc_results

            print(f"  VaR: {mc_results['VaR']:.4%}")
            print(f"  CVaR: {mc_results['CVaR']:.4%}")
            print(f"  ES: {mc_results['ES']:.4%}")
            print(f"  预期损失: {mc_results['expected_loss']:.4%}")
            print(f"  预期收益: {mc_results['expected_gain']:.4%}")

        # 4. 预期损失 (Expected Shortfall)
        print("\n预期损失 (ES):")
        es_results = self._calculate_expected_shortfall(returns, alpha=alpha)
        results['expected_shortfall'] = es_results

        for confidence, es in es_results.items():
            print(f"  {confidence}% ES: {es:.4%}")

        self.risk_metrics = results
        return results

    def _calculate_parametric_var(self, returns, alpha=0.95, distribution='normal'):
        """参数法计算VaR"""
        if distribution == 'normal':
            mu = returns.mean()
            sigma = returns.std()
            var = mu + sigma * stats.norm.ppf(1 - alpha)
        elif distribution == 't':
            # 拟合t分布
            df, loc, scale = stats.t.fit(returns)
            var = stats.t.ppf(1 - alpha, df, loc, scale)
        else:
            raise ValueError("分布类型必须是 'normal' 或 't'")

        return var

    def _calculate_parametric_cvar(self, returns, alpha=0.95, distribution='normal'):
        """参数法计算CVaR"""
        if distribution == 'normal':
            mu = returns.mean()
            sigma = returns.std()
            z_alpha = stats.norm.ppf(alpha)
            cvar = mu - sigma * stats.norm.pdf(z_alpha) / (1 - alpha)
        elif distribution == 't':
            df, loc, scale = stats.t.fit(returns)
            x_alpha = stats.t.ppf(alpha, df, loc, scale)
            cvar = - (loc + scale * (df + x_alpha ** 2) / (df - 1) *
                      stats.t.pdf(x_alpha, df) / (1 - alpha))
        else:
            raise ValueError("分布类型必须是 'normal' 或 't'")

        return cvar

    def _calculate_var_horizon(self, returns, alpha=0.95, horizon=10, method='historical'):
        """计算多期VaR"""
        if method == 'historical':
            # 历史模拟法
            horizon_returns = returns.rolling(horizon).sum().dropna()
            return np.percentile(horizon_returns, (1 - alpha) * 100)
        elif method == 'parametric':
            # 参数法（正态分布）
            mu = returns.mean() * horizon
            sigma = returns.std() * np.sqrt(horizon)
            return mu + sigma * stats.norm.ppf(1 - alpha)
        else:
            return None

    def _calculate_cvar_horizon(self, returns, alpha=0.95, horizon=10, method='historical'):
        """计算多期CVaR"""
        if method == 'historical':
            horizon_returns = returns.rolling(horizon).sum().dropna()
            var = np.percentile(horizon_returns, (1 - alpha) * 100)
            return horizon_returns[horizon_returns <= var].mean()
        else:
            return None

    def _calculate_expected_shortfall(self, returns, confidence_levels=[0.95, 0.99, 0.995]):
        """计算预期损失"""
        es_results = {}

        for confidence in confidence_levels:
            var = np.percentile(returns, (1 - confidence) * 100)
            es = returns[returns <= var].mean()
            es_results[f'ES_{int(confidence * 100)}'] = es

        return es_results

    def _monte_carlo_var(self, alpha=0.95, horizon=10, n_simulations=10000,
                         method='geometric_brownian'):
        """蒙特卡洛模拟计算VaR"""
        print(f"\n蒙特卡洛模拟: {n_simulations} 次模拟, {horizon} 天持有期")

        if self.portfolio_returns is None:
            print("错误: 没有投资组合数据")
            return None

        returns = self.portfolio_returns.dropna()
        log_returns = self.portfolio_log_returns.dropna()

        if method == 'geometric_brownian':
            # 几何布朗运动
            mu = log_returns.mean()
            sigma = log_returns.std()
            S0 = 100  # 初始投资额

            simulated_pnl = []

            for _ in tqdm(range(n_simulations), desc="蒙特卡洛模拟"):
                # 生成随机路径
                z = np.random.normal(0, 1, horizon)
                price_path = S0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) + sigma * z))
                horizon_return = (price_path[-1] - S0) / S0
                simulated_pnl.append(horizon_return)

        elif method == 'historical_bootstrap':
            # 历史自举法
            simulated_pnl = []

            for _ in tqdm(range(n_simulations), desc="历史自举法"):
                # 从历史数据中随机抽样
                sample_returns = np.random.choice(returns, size=horizon, replace=True)
                horizon_return = np.prod(1 + sample_returns) - 1
                simulated_pnl.append(horizon_return)

        else:
            raise ValueError("方法必须是 'geometric_brownian' 或 'historical_bootstrap'")

        simulated_pnl = np.array(simulated_pnl)

        # 计算风险指标
        var = np.percentile(simulated_pnl, (1 - alpha) * 100)
        cvar = simulated_pnl[simulated_pnl <= var].mean()
        es = simulated_pnl[simulated_pnl <= var].mean()  # ES = CVaR

        expected_loss = simulated_pnl[simulated_pnl < 0].mean() if (simulated_pnl < 0).any() else 0
        expected_gain = simulated_pnl[simulated_pnl > 0].mean() if (simulated_pnl > 0).any() else 0

        # 计算VaR分解
        var_decomposition = self._calculate_var_decomposition(alpha, horizon)

        results = {
            'VaR': var,
            'CVaR': cvar,
            'ES': es,
            'expected_loss': expected_loss,
            'expected_gain': expected_gain,
            'simulated_returns': simulated_pnl,
            'VaR_decomposition': var_decomposition
        }

        # 可视化结果
        self._plot_monte_carlo_results(simulated_pnl, var, cvar, alpha, horizon)

        return results

    def _calculate_var_decomposition(self, alpha=0.95, horizon=10):
        """计算VaR分解（边际VaR、成分VaR、增量VaR）"""
        if self.returns is None or self.portfolio_weights is None:
            return None

        returns_matrix = self.returns.values
        weights = self.portfolio_weights
        n_assets = len(weights)

        # 计算协方差矩阵
        cov_matrix = np.cov(returns_matrix, rowvar=False) * 252

        # 计算组合波动率
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

        # 边际VaR
        marginal_var = (cov_matrix @ weights) / portfolio_vol * stats.norm.ppf(alpha)

        # 成分VaR
        component_var = weights * marginal_var

        # 增量VaR（近似）
        incremental_var = []
        for i in range(n_assets):
            # 暂时移除资产i
            temp_weights = weights.copy()
            temp_weights[i] = 0
            temp_weights = temp_weights / temp_weights.sum()
            temp_vol = np.sqrt(temp_weights.T @ cov_matrix @ temp_weights)
            temp_var = temp_vol * stats.norm.ppf(alpha)

            # 计算增量
            base_var = portfolio_vol * stats.norm.ppf(alpha)
            inc_var = base_var - temp_var
            incremental_var.append(inc_var)

        var_decomposition = {
            'marginal_VaR': dict(zip(self.returns.columns, marginal_var)),
            'component_VaR': dict(zip(self.returns.columns, component_var)),
            'incremental_VaR': dict(zip(self.returns.columns, incremental_var)),
            'portfolio_VaR': portfolio_vol * stats.norm.ppf(alpha)
        }

        return var_decomposition

    def _plot_monte_carlo_results(self, simulated_returns, var, cvar, alpha, horizon):
        """可视化蒙特卡洛模拟结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 模拟收益分布
        axes[0, 0].hist(simulated_returns * 100, bins=50, density=True,
                        alpha=0.6, color='steelblue', edgecolor='black')

        # 添加正态分布拟合
        x = np.linspace(simulated_returns.min(), simulated_returns.max(), 100)
        mu, std = simulated_returns.mean(), simulated_returns.std()
        axes[0, 0].plot(x * 100, stats.norm.pdf(x, mu, std) * 100,
                        'r-', linewidth=2, label='正态分布拟合')

        # 标记VaR和CVaR
        axes[0, 0].axvline(x=var * 100, color='red', linestyle='--',
                           linewidth=2, label=f'VaR({alpha * 100:.0f}%)={var * 100:.2f}%')
        axes[0, 0].axvline(x=cvar * 100, color='darkred', linestyle='--',
                           linewidth=2, label=f'CVaR({alpha * 100:.0f}%)={cvar * 100:.2f}%')

        axes[0, 0].axvline(x=0, color='black', linestyle='-', alpha=0.5)

        # 填充尾部区域
        x_fill = np.linspace(simulated_returns.min(), var, 100)
        axes[0, 0].fill_between(x_fill * 100, 0, stats.norm.pdf(x_fill, mu, std) * 100,
                                color='red', alpha=0.3, label='尾部风险')

        axes[0, 0].set_xlabel(f'{horizon}天收益率 (%)')
        axes[0, 0].set_ylabel('概率密度')
        axes[0, 0].set_title(f'蒙特卡洛模拟收益分布 (n={len(simulated_returns)})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 模拟路径
        axes[0, 1].clear()
        n_paths_to_show = 20

        S0 = 100
        mu = simulated_returns.mean() / horizon
        sigma = simulated_returns.std() / np.sqrt(horizon)

        for i in range(min(n_paths_to_show, len(simulated_returns))):
            z = np.random.normal(0, 1, horizon)
            path = S0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) + sigma * z))
            axes[0, 1].plot(range(horizon), path, alpha=0.5, linewidth=0.8)

        # 计算分位数路径
        percentiles = [5, 25, 50, 75, 95]
        percentile_paths = []

        for perc in percentiles:
            percentile_returns = np.percentile(simulated_returns, perc)
            path = S0 * (1 + percentile_returns)
            axes[0, 1].axhline(y=path, color='black', linestyle='--', alpha=0.5,
                               label=f'{perc}%' if perc == 5 else "")

        axes[0, 1].axhline(y=S0, color='red', linestyle='-', linewidth=2, label='初始价值')
        axes[0, 1].set_xlabel('交易日')
        axes[0, 1].set_ylabel('投资组合价值')
        axes[0, 1].set_title('蒙特卡洛模拟价格路径示例')
        axes[0, 1].legend(loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 风险贡献
        if hasattr(self, 'risk_metrics') and 'monte_carlo' in self.risk_metrics:
            var_decomp = self.risk_metrics['monte_carlo'].get('VaR_decomposition')
            if var_decomp and 'component_VaR' in var_decomp:
                component_var = var_decomp['component_VaR']

                if isinstance(component_var, dict):
                    assets = list(component_var.keys())
                    contributions = list(component_var.values())

                    # 计算贡献度
                    total_var = abs(var_decomp.get('portfolio_VaR', 1))
                    contributions_pct = [abs(c) / total_var * 100 for c in contributions]

                    axes[1, 0].barh(range(len(assets)), contributions_pct)
                    axes[1, 0].set_yticks(range(len(assets)))
                    axes[1, 0].set_yticklabels(assets)
                    axes[1, 0].set_xlabel('成分VaR贡献 (%)')
                    axes[1, 0].set_title('成分VaR分解')
                    axes[1, 0].grid(True, alpha=0.3, axis='x')

        # 4. 统计摘要
        axes[1, 1].axis('off')
        stats_text = f"""
        蒙特卡洛模拟结果统计

        模拟次数: {len(simulated_returns)}
        持有期: {horizon} 天
        置信水平: {alpha * 100:.0f}%

        风险指标:
        - VaR({alpha * 100:.0f}%): {var * 100:.2f}%
        - CVaR({alpha * 100:.0f}%): {cvar * 100:.2f}%
        - 预期损失: {self.risk_metrics['monte_carlo']['expected_loss'] * 100:.2f}%
        - 预期收益: {self.risk_metrics['monte_carlo']['expected_gain'] * 100:.2f}%

        分布统计:
        - 均值: {simulated_returns.mean() * 100:.2f}%
        - 标准差: {simulated_returns.std() * 100:.2f}%
        - 偏度: {stats.skew(simulated_returns):.4f}
        - 峰度: {stats.kurtosis(simulated_returns):.4f}
        - 最小值: {simulated_returns.min() * 100:.2f}%
        - 最大值: {simulated_returns.max() * 100:.2f}%
        - 损失概率: {(simulated_returns < 0).mean() * 100:.1f}%
        """

        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, fontfamily='monospace',
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('monte_carlo_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def calculate_correlation_analysis(self):
        """计算相关性分析"""
        print("\n" + "=" * 60)
        print("3. 相关性分析")
        print("=" * 60)

        if self.returns is None:
            print("错误: 没有收益率数据")
            return

        # 计算滚动相关性
        window = 60
        rolling_corr = self.returns.rolling(window=window).corr()

        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 相关系数矩阵热图
        im = axes[0, 0].imshow(self.corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[0, 0].set_xticks(range(len(self.corr_matrix.columns)))
        axes[0, 0].set_yticks(range(len(self.corr_matrix.columns)))
        axes[0, 0].set_xticklabels(self.corr_matrix.columns, rotation=45, ha='right')
        axes[0, 0].set_yticklabels(self.corr_matrix.columns)
        axes[0, 0].set_title('资产相关系数矩阵')
        plt.colorbar(im, ax=axes[0, 0])

        # 添加相关系数数值
        for i in range(len(self.corr_matrix.columns)):
            for j in range(len(self.corr_matrix.columns)):
                text = axes[0, 0].text(j, i, f'{self.corr_matrix.iloc[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontsize=8)

        # 2. 滚动相关性
        if len(self.returns.columns) >= 2:
            asset1, asset2 = self.returns.columns[0], self.returns.columns[1]
            rolling_corr_pair = self.returns[asset1].rolling(window=window).corr(self.returns[asset2])

            axes[0, 1].plot(rolling_corr_pair.index, rolling_corr_pair.values, linewidth=1.5)
            axes[0, 1].axhline(y=rolling_corr_pair.mean(), color='red', linestyle='--',
                               label=f'均值: {rolling_corr_pair.mean():.3f}')
            axes[0, 1].fill_between(rolling_corr_pair.index,
                                    rolling_corr_pair.values,
                                    rolling_corr_pair.mean(),
                                    alpha=0.3, color='blue')
            axes[0, 1].set_xlabel('日期')
            axes[0, 1].set_ylabel('相关系数')
            axes[0, 1].set_title(f'{asset1} 和 {asset2} 滚动相关性 (窗口: {window}天)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 相关系数分布
        upper_tri = np.triu_indices_from(self.corr_matrix, k=1)
        correlation_values = self.corr_matrix.values[upper_tri]

        axes[1, 0].hist(correlation_values, bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=correlation_values.mean(), color='red', linestyle='--',
                           label=f'均值: {correlation_values.mean():.3f}')
        axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_xlabel('相关系数')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('资产相关系数分布')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 相关矩阵特征值分解
        eigenvals, eigenvecs = np.linalg.eig(self.corr_matrix.values)
        variance_explained = eigenvals / eigenvals.sum() * 100

        axes[1, 1].bar(range(1, len(variance_explained) + 1), variance_explained,
                       alpha=0.7, edgecolor='black')
        axes[1, 1].plot(range(1, len(variance_explained) + 1),
                        np.cumsum(variance_explained), 'ro-', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('主成分')
        axes[1, 1].set_ylabel('方差解释率 (%)')
        axes[1, 1].set_title('相关系数矩阵特征值分解')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xticks(range(1, len(variance_explained) + 1))

        # 添加累计方差标签
        for i, (v, c) in enumerate(zip(variance_explained, np.cumsum(variance_explained)), 1):
            axes[1, 1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=8)
            axes[1, 1].text(i, c + 1, f'{c:.1f}%', ha='center', fontsize=8, color='red')

        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 打印统计结果
        print("\n相关性分析结果:")
        print("-" * 40)
        print(f"平均相关系数: {correlation_values.mean():.4f}")
        print(f"相关系数标准差: {correlation_values.std():.4f}")
        print(f"最大相关系数: {correlation_values.max():.4f}")
        print(f"最小相关系数: {correlation_values.min():.4f}")
        print(f"正相关比例: {(correlation_values > 0).sum() / len(correlation_values):.1%}")

        # 计算第一主成分
        if len(eigenvals) > 0:
            first_pc = eigenvecs[:, 0]
            print(f"第一主成分解释方差: {variance_explained[0]:.1f}%")
            print(f"主成分载荷: {first_pc}")

        return self.corr_matrix

    def stress_testing(self, scenarios=None):
        """
        压力测试
        包含历史压力测试和假设情景分析
        """
        print("\n" + "=" * 60)
        print("4. 压力测试")
        print("=" * 60)

        if self.portfolio_returns is None:
            print("错误: 没有投资组合数据")
            return

        if scenarios is None:
            scenarios = {
                '历史压力': {
                    '2008金融危机': {'2008-01-01': 0.8, '2008-10-01': 0.7, '2009-03-01': 0.6},
                    '2015年股灾': {'2015-06-01': 0.7, '2015-08-01': 0.6, '2015-12-01': 0.8},
                    '2020疫情崩盘': {'2020-01-01': 0.9, '2020-03-01': 0.7, '2020-06-01': 0.8}
                },
                '假设情景': {
                    '温和衰退': {'股票': -0.2, '债券': 0.1, '商品': -0.1, '房地产': -0.15},
                    '严重衰退': {'股票': -0.4, '债券': 0.2, '商品': -0.2, '房地产': -0.3},
                    '利率上升': {'股票': -0.15, '债券': -0.2, '商品': -0.1, '房地产': -0.25},
                    '通胀上升': {'股票': -0.1, '债券': -0.3, '商品': 0.2, '房地产': 0.1}
                }
            }

        stress_test_results = {}

        # 1. 历史压力测试
        print("\n历史压力测试:")
        print("-" * 40)

        for scenario_name, stress_periods in scenarios['历史压力'].items():
            print(f"\n情景: {scenario_name}")

            # 计算压力期间损失
            max_drawdown = 0
            total_loss = 0
            stress_days = 0

            for period_start, stress_factor in stress_periods.items():
                if period_start in self.portfolio_returns.index:
                    # 找到压力期间的数据
                    start_idx = self.portfolio_returns.index.get_loc(period_start)
                    end_idx = min(start_idx + 60, len(self.portfolio_returns))

                    stress_returns = self.portfolio_returns.iloc[start_idx:end_idx]

                    if len(stress_returns) > 0:
                        # 计算期间收益率
                        period_return = (1 + stress_returns).prod() - 1
                        stressed_return = period_return * stress_factor

                        # 计算期间最大回撤
                        cumulative = (1 + stress_returns).cumprod()
                        rolling_max = cumulative.expanding().max()
                        drawdown = (cumulative - rolling_max) / rolling_max
                        period_dd = drawdown.min()

                        print(f"  {period_start} 开始: 收益率 {period_return:.2%} -> {stressed_return:.2%}")
                        print(f"  期间最大回撤: {period_dd:.2%}")

                        total_loss += stressed_return
                        max_drawdown = min(max_drawdown, period_dd)
                        stress_days += len(stress_returns)

            if stress_days > 0:
                avg_loss = total_loss
                stress_test_results[scenario_name] = {
                    '平均收益率': avg_loss,
                    '最大回撤': max_drawdown,
                    '天数': stress_days
                }

        # 2. 假设情景分析
        print("\n假设情景分析:")
        print("-" * 40)

        portfolio_value = 100
        asset_classes = {'股票': 0.5, '债券': 0.3, '商品': 0.1, '房地产': 0.1}

        for scenario_name, shocks in scenarios['假设情景'].items():
            scenario_value = portfolio_value
            scenario_details = {}

            for asset_class, shock in shocks.items():
                class_value = portfolio_value * asset_classes[asset_class]
                shocked_value = class_value * (1 + shock)
                scenario_value += shocked_value - class_value

                scenario_details[asset_class] = {
                    '配置权重': asset_classes[asset_class],
                    '冲击': shock,
                    '冲击后权重': shocked_value / (scenario_value if scenario_value > 0 else 1)
                }

            loss_pct = (scenario_value - portfolio_value) / portfolio_value

            stress_test_results[scenario_name] = {
                '组合价值': scenario_value,
                '收益率': loss_pct,
                '情景详情': scenario_details
            }

            print(f"\n情景: {scenario_name}")
            print(f"  组合收益率: {loss_pct:.2%}")
            print(f"  组合价值变化: {portfolio_value:.1f} -> {scenario_value:.1f}")
            for asset, details in scenario_details.items():
                print(f"  {asset}: 权重 {details['配置权重']:.1%}, 冲击 {details['冲击']:+.1%}, "
                      f"新权重 {details['冲击后权重']:.1%}")

        # 可视化压力测试结果
        self._plot_stress_test(stress_test_results)

        self.stress_test_results = stress_test_results
        return stress_test_results

    def _plot_stress_test(self, stress_test_results):
        """可视化压力测试结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 历史压力测试结果
        historical_scenarios = {k: v for k, v in stress_test_results.items()
                                if '收益率' in v and isinstance(v['收益率'], (int, float, np.number))}

        if historical_scenarios:
            scenario_names = list(historical_scenarios.keys())
            returns = [historical_scenarios[s]['收益率'] for s in scenario_names]

            colors = ['red' if r < 0 else 'green' for r in returns]
            axes[0, 0].bar(scenario_names, returns, color=colors, alpha=0.7, edgecolor='black')

            axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[0, 0].set_xlabel('压力情景')
            axes[0, 0].set_ylabel('收益率 (%)')
            axes[0, 0].set_title('历史压力测试情景收益')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for i, v in enumerate(returns):
                axes[0, 0].text(i, v / 2, f'{v:.1%}', ha='center', va='center',
                                color='white' if abs(v) > 0.1 else 'black', fontweight='bold')

        # 2. 假设情景分析
        hypothetical_scenarios = {k: v for k, v in stress_test_results.items()
                                  if '组合价值' in v and '情景详情' in v}

        if hypothetical_scenarios:
            scenario_names = list(hypothetical_scenarios.keys())
            portfolio_values = [hypothetical_scenarios[s]['组合价值'] for s in scenario_names]

            axes[0, 1].bar(scenario_names, portfolio_values, color='skyblue',
                           alpha=0.7, edgecolor='black')

            axes[0, 1].axhline(y=100, color='red', linestyle='--', label='初始价值 (100)')
            axes[0, 1].set_xlabel('压力情景')
            axes[0, 1].set_ylabel('组合价值')
            axes[0, 1].set_title('假设情景分析: 组合价值变化')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for i, v in enumerate(portfolio_values):
                axes[0, 1].text(i, v, f'{v:.1f}', ha='center', va='bottom')

        # 3. 热力图
        if hypothetical_scenarios:
            scenarios = list(hypothetical_scenarios.keys())
            assets = list(next(iter(hypothetical_scenarios.values()))['情景详情'].keys())

            shock_matrix = np.zeros((len(scenarios), len(assets)))

            for i, scenario in enumerate(scenarios):
                details = hypothetical_scenarios[scenario]['情景详情']
                for j, asset in enumerate(assets):
                    if asset in details:
                        shock_matrix[i, j] = details[asset]['冲击'] * 100

            im = axes[1, 0].imshow(shock_matrix, cmap='RdBu_r', aspect='auto')

            axes[1, 0].set_xticks(range(len(assets)))
            axes[1, 0].set_yticks(range(len(scenarios)))
            axes[1, 0].set_xticklabels(assets, rotation=45)
            axes[1, 0].set_yticklabels(scenarios)
            axes[1, 0].set_title('资产类别冲击矩阵 (%)')
            plt.colorbar(im, ax=axes[1, 0])

            # 添加数值
            for i in range(len(scenarios)):
                for j in range(len(assets)):
                    text = axes[1, 0].text(j, i, f'{shock_matrix[i, j]:.1f}',
                                           ha="center", va="center", color="black", fontsize=8)

        # 4. 冲击分解
        axes[1, 1].axis('off')

        if hypothetical_scenarios:
            scenario_name = list(hypothetical_scenarios.keys())[0]
            scenario_data = hypothetical_scenarios[scenario_name]

            if '情景详情' in scenario_data:
                details_text = f"情景: {scenario_name}\n\n"
                details_text += f"初始价值: 100.0\n"
                details_text += f"冲击后价值: {scenario_data['组合价值']:.1f}\n"
                details_text += f"收益率: {scenario_data['收益率']:.2%}\n\n"
                details_text += "资产类别冲击分解:\n"

                for asset, detail in scenario_data['情景详情'].items():
                    details_text += f"{asset}:\n"
                    details_text += f"  初始权重: {detail['配置权重']:.1%}\n"
                    details_text += f"  冲击: {detail['冲击']:+.1%}\n"
                    details_text += f"  冲击后权重: {detail['冲击后权重']:.1%}\n\n"

                axes[1, 1].text(0.1, 0.5, details_text, fontsize=9, fontfamily='monospace',
                                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('stress_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def backtest_var(self, var_level=0.95, window=252, method='historical'):
        """
        回测VaR模型
        包括无条件覆盖检验、条件覆盖检验
        """
        print("\n" + "=" * 60)
        print("5. VaR模型回测检验")
        print("=" * 60)

        if self.portfolio_returns is None:
            print("错误: 没有投资组合数据")
            return

        returns = self.portfolio_returns
        n_returns = len(returns)

        var_series = []
        hit_sequence = []
        var_methods = {}

        # 使用不同方法计算VaR
        for i in tqdm(range(window, n_returns), desc="计算滚动VaR"):
            # 获取滚动窗口数据
            window_returns = returns.iloc[i - window:i]
            current_return = returns.iloc[i]

            # 计算历史VaR
            var_hist = np.percentile(window_returns, (1 - var_level) * 100)
            var_series.append(var_hist)

            # 检查是否例外
            is_exception = current_return < var_hist
            hit_sequence.append(1 if is_exception else 0)

        hit_sequence = np.array(hit_sequence)
        test_returns = returns.iloc[window:]

        # 计算回测统计量
        n_obs = len(test_returns)
        n_exceptions = hit_sequence.sum()
        exception_rate = n_exceptions / n_obs
        expected_exceptions = (1 - var_level) * n_obs

        print(f"回测统计:")
        print(f"  观测天数: {n_obs}")
        print(f"  预期例外数: {expected_exceptions:.1f}")
        print(f"  实际例外数: {n_exceptions}")
        print(f"  例外率: {exception_rate * 100:.2f}% (预期: {(1 - var_level) * 100:.1f}%)")

        # Kupiec检验（无条件覆盖检验）
        likelihood_ratio_uc, p_value_uc = self._kupiec_test(n_obs, n_exceptions, 1 - var_level)

        print(f"\nKupiec无条件覆盖检验:")
        print(f"  似然比统计量: {likelihood_ratio_uc:.4f}")
        print(f"  P值: {p_value_uc:.4f}")

        if p_value_uc < 0.05:
            print("  结论: 拒绝原假设，VaR模型不充分")
        else:
            print("  结论: 不能拒绝原假设，VaR模型充分")

        # Christoffersen检验（条件覆盖检验）
        cc_results = self._christoffersen_test(hit_sequence, 1 - var_level)

        print(f"\nChristoffersen条件覆盖检验:")
        print(f"  转移矩阵: 0→0: {cc_results['n00']}, 0→1: {cc_results['n01']}, "
              f"1→0: {cc_results['n10']}, 1→1: {cc_results['n11']}")
        print(f"  条件概率: π0={cc_results['pi0']:.4f}, π1={cc_results['pi1']:.4f}")
        print(f"  条件覆盖似然比: {cc_results['lr_conditional']:.4f}")
        print(f"  P值: {cc_results['p_value_cc']:.4f}")

        if cc_results['p_value_cc'] < 0.05:
            print("  结论: 拒绝原假设，VaR模型不充分")
        else:
            print("  结论: 不能拒绝原假设，VaR模型充分")

        # 计算VaR偏差
        var_series = np.array(var_series)
        exceptions = test_returns.values[hit_sequence == 1]
        var_breaches = exceptions - var_series[hit_sequence == 1]

        if len(var_breaches) > 0:
            mean_breach = var_breaches.mean()
            max_breach = var_breaches.min()
            print(f"\nVaR突破统计:")
            print(f"  平均突破幅度: {mean_breach:.4%}")
            print(f"  最大突破幅度: {max_breach:.4%}")

        # 可视化回测结果
        self._plot_var_backtest(test_returns, var_series, hit_sequence, var_level)

        backtest_results = {
            'n_observations': n_obs,
            'n_exceptions': n_exceptions,
            'exception_rate': exception_rate,
            'expected_exceptions': expected_exceptions,
            'kupiec_test': {'LR': likelihood_ratio_uc, 'p_value': p_value_uc},
            'christoffersen_test': cc_results,
            'var_series': var_series,
            'hit_sequence': hit_sequence
        }

        return backtest_results

    def _kupiec_test(self, n, x, p):
        """Kupiec回测检验（无条件覆盖）"""
        from scipy.stats import chi2

        if x == 0:
            likelihood_ratio = -2 * np.log(((1 - p) ** n) / ((1 - x / n) ** n))
        elif x == n:
            likelihood_ratio = -2 * np.log((p ** n) / ((x / n) ** n))
        else:
            likelihood_ratio = -2 * np.log(
                ((1 - p) ** (n - x) * p ** x) /
                ((1 - x / n) ** (n - x) * (x / n) ** x)
            )

        p_value = 1 - chi2.cdf(likelihood_ratio, 1)
        return likelihood_ratio, p_value

    def _christoffersen_test(self, hit_sequence, p):
        """Christoffersen条件覆盖检验"""
        from scipy.stats import chi2

        # 计算转移矩阵
        n00 = n01 = n10 = n11 = 0

        for i in range(1, len(hit_sequence)):
            if hit_sequence[i - 1] == 0 and hit_sequence[i] == 0:
                n00 += 1
            elif hit_sequence[i - 1] == 0 and hit_sequence[i] == 1:
                n01 += 1
            elif hit_sequence[i - 1] == 1 and hit_sequence[i] == 0:
                n10 += 1
            elif hit_sequence[i - 1] == 1 and hit_sequence[i] == 1:
                n11 += 1

        # 计算概率
        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0

        # 无条件覆盖检验
        n_obs = len(hit_sequence)
        n_exceptions = hit_sequence.sum()
        lr_uc, _ = self._kupiec_test(n_obs, n_exceptions, p)

        # 独立性检验
        lr_ind = -2 * np.log(
            ((1 - pi) ** (n00 + n10) * pi ** (n01 + n11)) /
            ((1 - pi0) ** n00 * pi0 ** n01 * (1 - pi1) ** n10 * pi1 ** n11)
        )

        # 条件覆盖检验
        lr_cc = lr_uc + lr_ind
        p_value_cc = 1 - chi2.cdf(lr_cc, 2)

        return {
            'transition_matrix': [[n00, n01], [n10, n11]],
            'probabilities': {'pi0': pi0, 'pi1': pi1, 'pi': pi},
            'lr_independence': lr_ind,
            'lr_conditional': lr_cc,
            'p_value_cc': p_value_cc
        }

    def _plot_var_backtest(self, returns, var_series, hit_sequence, var_level):
        """可视化VaR回测结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. VaR与实际收益
        axes[0, 0].plot(returns.index, returns.values, 'b-', linewidth=0.5, alpha=0.7, label='实际收益')
        axes[0, 0].plot(returns.index, var_series, 'r-', linewidth=1, label=f'VaR({var_level * 100:.0f}%)')

        # 标记例外点
        exception_indices = np.where(hit_sequence == 1)[0]
        if len(exception_indices) > 0:
            exception_dates = returns.index[exception_indices]
            exception_returns = returns.values[exception_indices]
            axes[0, 0].scatter(exception_dates, exception_returns,
                               color='red', s=20, label='VaR例外点')

        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 0].set_xlabel('日期')
        axes[0, 0].set_ylabel('收益率')
        axes[0, 0].set_title('VaR回测: 实际收益 vs VaR')
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 例外序列
        axes[0, 1].plot(hit_sequence, 'b-', linewidth=0.5, drawstyle='steps')
        axes[0, 1].axhline(y=1 - var_level, color='red', linestyle='--',
                           label=f'预期例外率: {(1 - var_level) * 100:.1f}%')
        axes[0, 1].axhline(y=hit_sequence.mean(), color='green', linestyle='--',
                           label=f'实际例外率: {hit_sequence.mean() * 100:.2f}%')
        axes[0, 1].set_xlabel('观测点')
        axes[0, 1].set_ylabel('例外指示器 (0/1)')
        axes[0, 1].set_title('例外序列')
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(-0.1, 1.1)

        # 3. 例外分布
        returns_hist = returns.values
        var_hist = var_series

        axes[1, 0].hist(returns_hist, bins=50, density=True, alpha=0.6,
                        color='blue', edgecolor='black', label='收益分布')
        axes[1, 0].axvline(x=np.mean(var_hist), color='red', linestyle='--',
                           linewidth=2, label=f'平均VaR: {np.mean(var_hist):.4f}')

        # 标记例外区域
        if len(exception_indices) > 0:
            axes[1, 0].hist(exception_returns, bins=20, density=True,
                            alpha=0.8, color='red', edgecolor='darkred',
                            label='例外点分布')

        # 添加密度估计
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(returns_hist)
        x_range = np.linspace(returns_hist.min(), returns_hist.max(), 100)
        axes[1, 0].plot(x_range, kde(x_range), 'g-', linewidth=1.5, alpha=0.7, label='核密度估计')

        axes[1, 0].set_xlabel('收益率')
        axes[1, 0].set_ylabel('概率密度')
        axes[1, 0].set_title('收益率分布与VaR例外')
        axes[1, 0].legend(loc='upper right')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 回测统计
        axes[1, 1].axis('off')

        stats_text = f"""回测统计结果:

置信水平: {var_level * 100:.0f}%
观察窗口: 252天
观测天数: {len(returns)}
VaR计算方法: 历史模拟法

例外统计:
预期例外数: {(1 - var_level) * len(returns):.1f}
实际例外数: {hit_sequence.sum()}
例外率: {hit_sequence.mean() * 100:.2f}%

Kupiec检验:
统计量: {self._kupiec_test(len(returns), hit_sequence.sum(), 1 - var_level)[0]:.4f}
P值: {self._kupiec_test(len(returns), hit_sequence.sum(), 1 - var_level)[1]:.4f}

VaR突破统计:
平均突破幅度: {np.mean(exception_returns - var_hist[exception_indices]):.4%}
最大突破幅度: {np.min(exception_returns - var_hist[exception_indices]):.4%}
        """

        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=9, fontfamily='monospace',
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('var_backtest.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def calculate_portfolio_risk_decomposition(self):
        """计算投资组合风险分解"""
        print("\n" + "=" * 60)
        print("6. 投资组合风险分解")
        print("=" * 60)

        if self.returns is None or self.portfolio_weights is None:
            print("错误: 没有收益数据或投资组合权重")
            return

        # 计算协方差矩阵
        cov_matrix = self.returns.cov().values * 252
        weights = np.array(self.portfolio_weights)

        # 计算投资组合方差
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)

        # 计算边际风险贡献
        marginal_risk = (cov_matrix @ weights) / portfolio_volatility

        # 计算成分风险贡献
        component_risk = weights * marginal_risk

        # 计算增量风险贡献
        incremental_risk = []
        for i in range(len(weights)):
            # 临时移除第i个资产
            temp_weights = weights.copy()
            temp_weights[i] = 0
            if temp_weights.sum() > 0:
                temp_weights = temp_weights / temp_weights.sum()
            else:
                temp_weights = np.zeros_like(weights)

            temp_portfolio_variance = temp_weights.T @ cov_matrix @ temp_weights
            temp_portfolio_volatility = np.sqrt(temp_portfolio_variance)

            incremental_vol = portfolio_volatility - temp_portfolio_volatility
            incremental_risk.append(incremental_vol)

        # 计算集中度指标
        herfindahl_index = np.sum(weights ** 2)
        gini_coefficient = self._calculate_gini_coefficient(component_risk)

        # 创建结果DataFrame
        risk_decomp = pd.DataFrame({
            '资产': self.returns.columns,
            '权重': weights,
            '年化波动率': np.diag(cov_matrix) ** 0.5,
            '边际风险贡献': marginal_risk,
            '成分风险贡献': component_risk,
            '风险贡献比例': component_risk / portfolio_volatility * 100,
            '增量风险': incremental_risk
        })

        print("\n投资组合风险分解:")
        print("-" * 60)
        print(f"投资组合年化波动率: {portfolio_volatility:.2%}")
        print(f"投资组合年化方差: {portfolio_variance:.4%}")
        print(f"赫芬达尔指数: {herfindahl_index:.4f}")
        print(f"风险贡献基尼系数: {gini_coefficient:.4f}")
        print("\n详细分解:")
        print(risk_decomp.round(4).to_string(index=False))

        # 可视化
        self._plot_risk_decomposition(risk_decomp, portfolio_volatility)

        return risk_decomp

    def _calculate_gini_coefficient(self, values):
        """计算基尼系数"""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        index = np.arange(1, n + 1)

        gini = (np.sum((2 * index - n - 1) * sorted_values)) / (n * np.sum(sorted_values))
        return gini

    def _plot_risk_decomposition(self, risk_decomp, portfolio_vol):
        """可视化风险分解"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 风险贡献饼图
        risk_contribution = risk_decomp['成分风险贡献'].abs()
        risk_contribution_pct = risk_contribution / risk_contribution.sum() * 100

        wedges, texts, autotexts = axes[0, 0].pie(risk_contribution,
                                                  labels=risk_decomp['资产'],
                                                  autopct='%1.1f%%',
                                                  startangle=90)
        axes[0, 0].set_title('投资组合风险贡献分解')

        # 2. 权重 vs 风险贡献
        axes[0, 1].scatter(risk_decomp['权重'] * 100, risk_decomp['风险贡献比例'], s=100, alpha=0.6)

        for i, row in risk_decomp.iterrows():
            axes[0, 1].annotate(row['资产'], (row['权重'] * 100, row['风险贡献比例']),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)

        axes[0, 1].plot([0, risk_decomp['权重'].max() * 100], [0, risk_decomp['风险贡献比例'].max()],
                        'r--', alpha=0.5, label='等比例线')
        axes[0, 1].set_xlabel('投资组合权重 (%)')
        axes[0, 1].set_ylabel('风险贡献比例 (%)')
        axes[0, 1].set_title('权重 vs 风险贡献')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 边际风险贡献
        x = np.arange(len(risk_decomp))
        width = 0.35

        axes[1, 0].bar(x - width / 2, risk_decomp['边际风险贡献'] * 100, width, label='边际风险贡献', alpha=0.8)
        axes[1, 0].bar(x + width / 2, risk_decomp['增量风险'] * 100, width, label='增量风险', alpha=0.8)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(risk_decomp['资产'], rotation=45)
        axes[1, 0].set_ylabel('贡献 (%)')
        axes[1, 0].set_title('边际风险贡献 vs 增量风险')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. 风险分解摘要
        axes[1, 1].axis('off')

        summary_text = f"""投资组合风险分解摘要

投资组合年化波动率: {portfolio_vol:.2%}
风险集中度指标:
- 赫芬达尔指数: {np.sum(risk_decomp['权重'] ** 2):.4f}
- 风险贡献基尼系数: {self._calculate_gini_coefficient(risk_decomp['成分风险贡献']):.4f}

风险分散性:
- 前两大资产风险贡献: {risk_contribution_pct.nlargest(2).sum():.1f}%
- 前三大资产风险贡献: {risk_contribution_pct.nlargest(3).sum():.1f}%

风险调整指标:
- 夏普比率: {(risk_decomp['权重'] @ risk_decomp['年化波动率']) / portfolio_vol:.3f}
- 风险回报比: {risk_decomp['权重'].sum() / portfolio_vol:.3f}
        """

        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=9, fontfamily='monospace',
                        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('risk_decomposition.png', dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def run_comprehensive_analysis(self, var_level=0.95, horizon=10, n_simulations=10000):
        """运行全面风险分析"""
        print("=" * 60)
        print("投资组合风险计量与蒙特卡洛模拟")
        print("=" * 60)

        # 1. 基本统计分析
        self._calculate_basic_statistics()
        # 2. 计算风险指标
        print("\n" + "=" * 60)
        print("计算主要风险指标...")
        print("=" * 60)

        risk_results = self.calculate_risk_metrics(
            alpha=var_level,
            horizon=horizon,
            methods=['historical', 'parametric', 'monte_carlo']
        )

        # 3. 相关性分析
        print("\n" + "=" * 60)
        print("执行相关性分析...")
        print("=" * 60)
        corr_analysis = self.calculate_correlation_analysis()

        # 4. 蒙特卡洛模拟
        print("\n" + "=" * 60)
        print("执行蒙特卡洛模拟...")
        print("=" * 60)
        mc_results = self._monte_carlo_var(
            alpha=var_level,
            horizon=horizon,
            n_simulations=n_simulations
        )

        # 5. 压力测试
        print("\n" + "=" * 60)
        print("执行压力测试...")
        print("=" * 60)
        stress_results = self.stress_testing()

        # 6. VaR回测
        print("\n" + "=" * 60)
        print("执行VaR模型回测...")
        print("=" * 60)
        backtest_results = self.backtest_var(var_level=var_level)

        # 7. 风险分解
        print("\n" + "=" * 60)
        print("执行投资组合风险分解...")
        print("=" * 60)
        risk_decomp = self.calculate_portfolio_risk_decomposition()

        # 8. 计算希腊字母（期权风险）
        print("\n" + "=" * 60)
        print("计算期权希腊字母...")
        print("=" * 60)
        greeks = self.calculate_greeks()

        # 9. 生成综合报告
        self.generate_comprehensive_report(
            risk_results=risk_results,
            mc_results=mc_results,
            stress_results=stress_results,
            backtest_results=backtest_results,
            risk_decomp=risk_decomp,
            greeks=greeks
        )

        return {
            'risk_metrics': risk_results,
            'monte_carlo': mc_results,
            'stress_test': stress_results,
            'var_backtest': backtest_results,
            'risk_decomposition': risk_decomp,
            'greeks': greeks
        }

    def calculate_greeks(self, S=100, K=100, T=1.0, r=0.02, sigma=0.2):
        """计算期权希腊字母"""
        from scipy.stats import norm

        def black_scholes(S, K, T, r, sigma, option_type='call'):
            """Black-Scholes期权定价"""
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

            return price, d1, d2

        def calculate_all_greeks(S, K, T, r, sigma, option_type='call'):
            """计算所有希腊字母"""
            price, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)

            if option_type == 'call':
                delta = norm.cdf(d1)
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                vega = S * norm.pdf(d1) * np.sqrt(T)  # 每1%波动率变化
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                         r * K * np.exp(-r * T) * norm.cdf(d2)) / 365  # 每日theta
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)  # 每1%利率变化
            else:  # put
                delta = norm.cdf(d1) - 1
                gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
                vega = S * norm.pdf(d1) * np.sqrt(T)
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                         r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

            return {
                'Price': price,
                'Delta': delta,
                'Gamma': gamma,
                'Vega': vega,
                'Theta': theta,
                'Rho': rho
            }

        # 计算看涨和看跌期权的希腊字母
        call_greeks = calculate_all_greeks(S, K, T, r, sigma, 'call')
        put_greeks = calculate_all_greeks(S, K, T, r, sigma, 'put')

        print("\n期权希腊字母计算:")
        print("-" * 40)
        print(f"假设参数:")
        print(f"  标的价格: {S}")
        print(f"  行权价: {K}")
        print(f"  到期时间: {T} 年")
        print(f"  无风险利率: {r * 100:.1f}%")
        print(f"  波动率: {sigma * 100:.1f}%")

        print(f"\n看涨期权:")
        for greek, value in call_greeks.items():
            if greek in ['Delta', 'Gamma']:
                print(f"  {greek}: {value:.4f}")
            elif greek == 'Price':
                print(f"  {greek}: {value:.2f}")
            else:
                print(f"  {greek}: {value:.6f}")

        print(f"\n看跌期权:")
        for greek, value in put_greeks.items():
            if greek in ['Delta', 'Gamma']:
                print(f"  {greek}: {value:.4f}")
            elif greek == 'Price':
                print(f"  {greek}: {value:.2f}")
            else:
                print(f"  {greek}: {value:.6f}")

        # 可视化希腊字母敏感性
        self._plot_greeks_sensitivity(S, K, T, r, sigma)

        return {'call': call_greeks, 'put': put_greeks}

    def _plot_greeks_sensitivity(self, S, K, T, r, sigma):
        """可视化希腊字母敏感性"""
        from scipy.stats import norm

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 分析不同参数对希腊字母的影响
        parameters = {
            '标的价格': np.linspace(S * 0.5, S * 1.5, 50),
            '波动率': np.linspace(0.1, 0.5, 50),
            '到期时间': np.linspace(0.1, 2, 50),
            '行权价': np.linspace(K * 0.5, K * 1.5, 50),
            '无风险利率': np.linspace(0, 0.1, 50)
        }

        greek_names = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']

        for i, (param_name, param_values) in enumerate(parameters.items()):
            if i >= 6:  # 最多显示6个子图
                break

            ax = axes[i // 3, i % 3]
            greek_values = {greek: [] for greek in greek_names}

            for param_value in param_values:
                # 根据参数调整输入
                if param_name == '标的价格':
                    S_temp = param_value
                    sigma_temp = sigma
                    T_temp = T
                    r_temp = r
                    K_temp = K
                elif param_name == '波动率':
                    S_temp = S
                    sigma_temp = param_value
                    T_temp = T
                    r_temp = r
                    K_temp = K
                elif param_name == '到期时间':
                    S_temp = S
                    sigma_temp = sigma
                    T_temp = param_value
                    r_temp = r
                    K_temp = K
                elif param_name == '行权价':
                    S_temp = S
                    sigma_temp = sigma
                    T_temp = T
                    r_temp = r
                    K_temp = param_value
                else:  # 无风险利率
                    S_temp = S
                    sigma_temp = sigma
                    T_temp = T
                    r_temp = param_value
                    K_temp = K

                # 计算希腊字母
                d1 = (np.log(S_temp / K_temp) + (r_temp + 0.5 * sigma_temp ** 2) * T_temp) / (
                            sigma_temp * np.sqrt(T_temp))

                greek_values['Delta'].append(norm.cdf(d1))
                greek_values['Gamma'].append(norm.pdf(d1) / (S_temp * sigma_temp * np.sqrt(T_temp)))
                greek_values['Vega'].append(S_temp * norm.pdf(d1) * np.sqrt(T_temp))
                greek_values['Theta'].append((-S_temp * norm.pdf(d1) * sigma_temp / (2 * np.sqrt(T_temp)) -
                                              r_temp * K_temp * np.exp(-r_temp * T_temp) * norm.cdf(
                            d1 - sigma_temp * np.sqrt(T_temp))) / 365)
                greek_values['Rho'].append(
                    K_temp * T_temp * np.exp(-r_temp * T_temp) * norm.cdf(d1 - sigma_temp * np.sqrt(T_temp)))

            # 绘制敏感性
            for greek in greek_names:
                ax.plot(param_values, greek_values[greek], label=greek, linewidth=1.5)

            ax.set_xlabel(param_name)
            ax.set_ylabel('希腊字母值')
            ax.set_title(f'希腊字母 vs {param_name}')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('greeks_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_comprehensive_report(self, **results):
        """生成综合风险报告"""
        print("\n" + "=" * 60)
        print("生成综合风险报告")
        print("=" * 60)

        # 创建报告文本
        report = []
        report.append("=" * 60)
        report.append("投资组合风险计量综合报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"资产数量: {len(self.symbols)}")
        report.append(f"样本期间: {self.returns.index[0].date()} 到 {self.returns.index[-1].date()}")
        report.append(f"交易日数: {len(self.returns)}")
        report.append("")

        # 1. 投资组合概览
        report.append("1. 投资组合概览")
        report.append("-" * 40)
        for i, symbol in enumerate(self.symbols):
            report.append(f"  {symbol}: 权重 {self.portfolio_weights[i]:.1%}")

        if hasattr(self, 'portfolio_returns'):
            portfolio_stats = {
                '年化收益率': self.portfolio_returns.mean() * 252,
                '年化波动率': self.portfolio_returns.std() * np.sqrt(252),
                '夏普比率': (self.portfolio_returns.mean() * 252 - 0.02) / (
                            self.portfolio_returns.std() * np.sqrt(252)),
                '最大回撤': self._calculate_max_drawdown(self.portfolio_returns)
            }

            for key, value in portfolio_stats.items():
                if '率' in key or '撤' in key:
                    report.append(f"  {key}: {value:.2%}")
                else:
                    report.append(f"  {key}: {value:.3f}")

        # 2. 风险指标总结
        if 'risk_metrics' in results:
            report.append("\n2. 风险指标总结")
            report.append("-" * 40)

            risk_metrics = results['risk_metrics']
            for method, metrics in risk_metrics.items():
                if isinstance(metrics, dict):
                    report.append(f"  {method.upper()}方法:")
                    if 'VaR' in metrics and 'CVaR' in metrics:
                        report.append(f"    1天 95% VaR: {metrics['VaR']:.2%}")
                        report.append(f"    1天 95% CVaR: {metrics['CVaR']:.2%}")
                    elif isinstance(metrics, dict) and 'normal' in metrics:
                        report.append(f"    正态分布 VaR: {metrics['normal']['VaR']:.2%}")
                        report.append(f"    正态分布 CVaR: {metrics['normal']['CVaR']:.2%}")
                        report.append(f"    t分布 VaR: {metrics['t']['VaR']:.2%}")
                        report.append(f"    t分布 CVaR: {metrics['t']['CVaR']:.2%}")

        # 3. 蒙特卡洛模拟结果
        if 'monte_carlo' in results:
            mc = results['monte_carlo']
            report.append("\n3. 蒙特卡洛模拟结果")
            report.append("-" * 40)
            report.append(f"  模拟次数: {len(mc.get('simulated_returns', []))}")
            report.append(f"  VaR(95%): {mc.get('VaR', 0):.2%}")
            report.append(f"  CVaR(95%): {mc.get('CVaR', 0):.2%}")
            report.append(f"  预期损失: {mc.get('expected_loss', 0):.2%}")
            report.append(f"  预期收益: {mc.get('expected_gain', 0):.2%}")

        # 4. 压力测试结果
        if 'stress_test' in results:
            stress = results['stress_test']
            report.append("\n4. 压力测试结果")
            report.append("-" * 40)

            for scenario, data in stress.items():
                if isinstance(data, dict) and '收益率' in data:
                    report.append(f"  {scenario}: {data['收益率']:.2%}")

        # 5. VaR回测结果
        if 'var_backtest' in results:
            backtest = results['var_backtest']
            report.append("\n5. VaR模型回测")
            report.append("-" * 40)
            report.append(f"  观测天数: {backtest.get('n_observations', 0)}")
            report.append(f"  实际例外数: {backtest.get('n_exceptions', 0)}")
            report.append(f"  例外率: {backtest.get('exception_rate', 0):.2%}")

            kupiec = backtest.get('kupiec_test', {})
            if kupiec and 'p_value' in kupiec:
                report.append(f"  Kupiec检验P值: {kupiec['p_value']:.4f}")
                if kupiec['p_value'] < 0.05:
                    report.append("  Kupiec检验: 拒绝原假设")
                else:
                    report.append("  Kupiec检验: 接受原假设")

        # 6. 风险分解
        if 'risk_decomposition' in results:
            risk_decomp = results['risk_decomposition']
            if hasattr(risk_decomp, 'iloc'):
                report.append("\n6. 风险分解总结")
                report.append("-" * 40)
                report.append(f"  投资组合波动率: {risk_decomp['成分风险贡献'].sum():.2%}")
                report.append(f"  最大风险贡献资产: {risk_decomp.loc[risk_decomp['风险贡献比例'].idxmax(), '资产']}")
                report.append(f"  最大风险贡献比例: {risk_decomp['风险贡献比例'].max():.1f}%")

        # 7. 希腊字母
        if 'greeks' in results:
            greeks = results['greeks']
            report.append("\n7. 期权风险指标（示例）")
            report.append("-" * 40)
            if 'call' in greeks:
                call = greeks['call']
                report.append(f"  看涨期权Delta: {call.get('Delta', 0):.3f}")
                report.append(f"  看涨期权Gamma: {call.get('Gamma', 0):.4f}")
                report.append(f"  看涨期权Vega: {call.get('Vega', 0):.3f}")

        # 8. 风险管理建议
        report.append("\n8. 风险管理建议")
        report.append("-" * 40)

        # 根据结果生成建议
        suggestions = []

        # VaR相关建议
        if 'risk_metrics' in results:
            var_values = []
            for method, metrics in results['risk_metrics'].items():
                if isinstance(metrics, dict) and 'VaR' in metrics:
                    var_values.append(metrics['VaR'])

            if var_values:
                avg_var = np.mean(var_values)
                if avg_var < -0.05:
                    suggestions.append(f"  • 投资组合VaR较高({avg_var:.1%})，建议降低风险暴露")
                else:
                    suggestions.append(f"  • 当前VaR水平({avg_var:.1%})在可接受范围内")

        # 回测相关建议
        if 'var_backtest' in results:
            backtest = results['var_backtest']
            exception_rate = backtest.get('exception_rate', 0)
            expected_rate = 0.05  # 95%置信度

            if exception_rate > expected_rate * 1.5:
                suggestions.append(f"  • VaR模型低估风险(例外率{exception_rate:.1%} > 预期{expected_rate:.1%})")
            elif exception_rate < expected_rate * 0.5:
                suggestions.append(f"  • VaR模型可能过于保守")
            else:
                suggestions.append(f"  • VaR模型表现良好")

        # 风险分散建议
        if 'risk_decomposition' in results:
            risk_decomp = results['risk_decomposition']
            if hasattr(risk_decomp, 'iloc'):
                top3_risk = risk_decomp['风险贡献比例'].nlargest(3).sum()
                if top3_risk > 80:
                    suggestions.append(f"  • 风险高度集中(前3大资产贡献{top3_risk:.0f}%风险)")
                elif top3_risk > 60:
                    suggestions.append(f"  • 风险较为集中(前3大资产贡献{top3_risk:.0f}%风险)")
                else:
                    suggestions.append(f"  • 风险分散良好(前3大资产贡献{top3_risk:.0f}%风险)")

        if not suggestions:
            suggestions.append("  • 建议定期监控投资组合风险指标")
            suggestions.append("  • 建议定期进行压力测试")
            suggestions.append("  • 建议建立动态风险限额体系")

        report.extend(suggestions)

        # 9. 后续步骤
        report.append("\n9. 后续步骤")
        report.append("-" * 40)
        report.append("  • 监控每日VaR突破情况")
        report.append("  • 定期更新风险模型参数")
        report.append("  • 建立风险预警机制")
        report.append("  • 定期进行压力测试和情景分析")

        report.append("\n" + "=" * 60)

        # 保存报告
        report_text = "\n".join(report)

        # 保存到文件
        with open('risk_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n报告已保存到: risk_analysis_report.txt")

        return report_text


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    print("=" * 60)
    print("项目三：投资组合风险计量与蒙特卡洛模拟")
    print("=" * 60)

    # 初始化风险管理系统
    print("\n初始化风险管理系统...")
    risk_manager = PortfolioRiskManager(
        symbols=['000300.SH', '000905.SH', '000016.SH', '511010.SH', '518880.SH'],
        portfolio_weights=[0.4, 0.3, 0.1, 0.1, 0.1]  # 沪深300, 中证500, 上证50, 国债ETF, 黄金ETF
    )

    # 获取数据
    print("\n获取市场数据...")
    risk_manager.fetch_market_data(
        start_date='2019-01-01',
        end_date='2023-12-31',
        use_real_data=False,  # 使用模拟数据
        data_source='akshare'
    )

    # 运行全面分析
    print("\n执行全面风险分析...")
    results = risk_manager.run_comprehensive_analysis(
        var_level=0.95,
        horizon=10,
        n_simulations=5000
    )

    print("\n" + "=" * 60)
    print("项目完成总结")
    print("=" * 60)
    print("1. 实现了完整的风险计量框架，包含多种VaR/CVaR计算方法")
    print("2. 完成了蒙特卡洛模拟、压力测试、VaR回测等高级功能")
    print("3. 实现了投资组合风险分解和希腊字母计算")
    print("4. 生成了包含风险管理建议的综合报告")
    print("5. 所有结果已保存为图表和文本文件")
    print("=" * 60)

    return risk_manager, results


if __name__ == "__main__":
    # 运行主程序

    risk_manager, results = main()
