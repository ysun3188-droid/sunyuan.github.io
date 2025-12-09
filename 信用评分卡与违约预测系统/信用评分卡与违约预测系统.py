"""
================================================================================
项目名称
：信用评分卡系统 - 完整生产级实现
模块：数据生成、特征工程、模型训练、API部署、监控系统
================================================================================
"""
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import yaml
from typing import Dict, List, Any, Optional, Union
import logging
from pathlib import Path
import hashlib
import joblib

plt.style.use('seaborn-v0_8-darkgrid')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 模块一：配置管理
# ============================================================================

class ConfigManager:
    """配置管理器 - 统一管理所有配置"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'data': {
                'n_samples': 10000,
                'data_type': 'personal',  # personal, smb, corporate
                'test_size': 0.3,
                'random_state': 42
            },
            'features': {
                'woe': {
                    'n_bins': 10,
                    'min_samples_bin': 0.05,
                    'iv_threshold': 0.02
                },
                'scaling': 'standard',  # standard, minmax, robust
                'imputation': 'knn'  # mean, median, knn
            },
            'model': {
                'algorithms': ['logistic', 'random_forest', 'lightgbm', 'xgboost'],
                'ensemble_method': 'voting',
                'hyperparam_tuning': True,
                'cross_validation': {
                    'n_splits': 5,
                    'random_state': 42
                }
            },
            'scoring': {
                'base_score': 600,
                'pdo': 20,
                'odds': 20
            },
            'api': {
                'host': '0.0.0.0',
                'port': 5000,
                'debug': False,
                'workers': 4
            },
            'monitoring': {
                'psi_threshold': 0.1,
                'auc_decay_threshold': 0.05,
                'check_interval_days': 7
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def save(self, path: str = None):
        """保存配置"""
        if path is None:
            path = self.config_path
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"配置保存到: {path}")


# ============================================================================
# 模块二：数据管道
# ============================================================================

class DataPipeline:
    """数据管道 - 处理数据生成、清洗、特征工程"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.data_generator = DataGenerator(config)
        self.feature_engineer = FeatureEngineer(config)
        self.preprocessor = None

    def run(self, mode: str = 'train') -> Dict:
        """运行数据管道"""
        logger.info(f"开始数据管道，模式: {mode}")

        # 1. 数据生成/加载
        if mode == 'train':
            data = self.data_generator.generate()
        else:
            # 在实际项目中，这里会从数据库或文件加载
            data = self.load_production_data()

        # 2. 特征工程
        processed_data = self.feature_engineer.transform(data)

        # 3. 数据拆分
        if mode == 'train':
            train_test_split = self.split_data(processed_data)
            return train_test_split
        else:
            return {'X': processed_data, 'raw_data': data}

    def split_data(self, data: pd.DataFrame) -> Dict:
        """拆分训练集和测试集"""
        from sklearn.model_selection import train_test_split

        X = data.drop('default', axis=1, errors='ignore')
        y = data['default'] if 'default' in data.columns else None

        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.get('data.test_size'),
                random_state=self.config.get('data.random_state'),
                stratify=y
            )

            logger.info(f"数据拆分: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")

            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'raw_data': data
            }

        return {'X': X, 'raw_data': data}

    def load_production_data(self) -> pd.DataFrame:
        """加载生产数据"""
        # 模拟加载生产数据
        logger.info("加载生产数据...")
        return self.data_generator.generate(n_samples=1000)

    def save_pipeline(self, path: str = 'models/pipeline.pkl'):
        """保存数据管道"""
        pipeline_data = {
            'feature_engineer': self.feature_engineer,
            'preprocessor': self.preprocessor
        }
        joblib.dump(pipeline_data, path)
        logger.info(f"数据管道保存到: {path}")

    def load_pipeline(self, path: str = 'models/pipeline.pkl'):
        """加载数据管道"""
        pipeline_data = joblib.load(path)
        self.feature_engineer = pipeline_data['feature_engineer']
        self.preprocessor = pipeline_data['preprocessor']
        logger.info(f"数据管道从 {path} 加载")


# ============================================================================
# 模块三：数据生成器
# ============================================================================

class DataGenerator:
    """数据生成器 - 生成模拟信用数据"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.data_type = config.get('data.data_type', 'personal')

    def generate(self, n_samples: int = None) -> pd.DataFrame:
        """生成数据"""
        if n_samples is None:
            n_samples = self.config.get('data.n_samples', 10000)

        logger.info(f"生成 {n_samples} 个 {self.data_type} 样本")

        if self.data_type == 'personal':
            return self._generate_personal_data(n_samples)
        elif self.data_type == 'smb':
            return self._generate_smb_data(n_samples)
        else:  # corporate
            return self._generate_corporate_data(n_samples)

    def _generate_personal_data(self, n_samples: int) -> pd.DataFrame:
        """生成个人贷款数据"""
        np.random.seed(self.config.get('data.random_state', 42))

        data = {}

        # 基本信息
        data['age'] = np.random.normal(40, 15, n_samples).clip(18, 80)
        data['gender'] = np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4])
        data['monthly_income'] = np.random.lognormal(9.5, 0.5, n_samples)
        data['employment_years'] = np.random.exponential(5, n_samples).clip(0, 40)
        data['credit_score'] = np.random.normal(650, 100, n_samples).clip(300, 850)

        # 贷款信息
        data['loan_amount'] = np.random.lognormal(10, 1, n_samples)
        data['loan_term'] = np.random.choice([12, 24, 36, 48, 60], n_samples)
        data['debt_to_income'] = np.random.beta(2, 8, n_samples)

        # 信用历史
        data['num_delinquencies'] = np.random.poisson(0.5, n_samples)
        data['num_credit_inquiries'] = np.random.poisson(2, n_samples)

        # 计算违约概率
        df = pd.DataFrame(data)
        df['default_probability'] = self._calculate_default_probability(df)
        df['default'] = np.random.binomial(1, df['default_probability'])

        # 添加时间戳和ID
        df['application_date'] = pd.date_range(
            start='2023-01-01',
            periods=n_samples,
            freq='D'
        )
        df['application_id'] = [f'APP_{i:08d}' for i in range(1, n_samples + 1)]

        return df

    def _calculate_default_probability(self, df: pd.DataFrame) -> pd.Series:
        """计算违约概率"""
        # 风险因子
        age_risk = np.where(df['age'] < 25, 0.1,
                            np.where(df['age'] < 40, 0.05,
                                     np.where(df['age'] < 60, 0.02, 0.08)))

        income_risk = np.where(df['monthly_income'] < 5000, 0.15,
                               np.where(df['monthly_income'] < 10000, 0.08,
                                        np.where(df['monthly_income'] < 20000, 0.03, 0.01)))

        credit_risk = np.where(df['credit_score'] < 500, 0.25,
                               np.where(df['credit_score'] < 600, 0.15,
                                        np.where(df['credit_score'] < 700, 0.05, 0.01)))

        dti_risk = np.where(df['debt_to_income'] > 0.5, 0.2,
                            np.where(df['debt_to_income'] > 0.3, 0.1, 0.05))

        # 组合风险
        base_risk = 0.05
        combined_risk = (age_risk + income_risk + credit_risk + dti_risk) / 4

        default_prob = base_risk + combined_risk
        default_prob = np.clip(default_prob, 0, 1)

        # 添加随机噪声
        default_prob += np.random.normal(0, 0.05, len(default_prob))
        default_prob = np.clip(default_prob, 0, 1)

        return default_prob

    def _generate_smb_data(self, n_samples: int) -> pd.DataFrame:
        """生成中小企业数据"""
        # 简化实现
        return self._generate_personal_data(n_samples)

    def _generate_corporate_data(self, n_samples: int) -> pd.DataFrame:
        """生成上市公司数据"""
        # 简化实现
        return self._generate_personal_data(n_samples)


# ============================================================================
# 模块四：特征工程
# ============================================================================

class FeatureEngineer:
    """特征工程 - 处理特征转换、编码、选择"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.woe_encoder = WOEEncoder(config)
        self.scaler = None
        self.imputer = None
        self.feature_names = []
        self.woe_info = {}

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换特征"""
        logger.info("开始特征工程...")

        # 1. 复制数据
        df = data.copy()

        # 2. 特征衍生
        df = self._create_features(df)

        # 3. 处理缺失值
        df = self._handle_missing_values(df)

        # 4. WOE编码
        if 'default' in df.columns:
            df = self.woe_encoder.fit_transform(df, target_col='default')
            self.woe_info = self.woe_encoder.woe_info
        else:
            df = self.woe_encoder.transform(df)

        # 5. 特征选择
        if 'default' in df.columns:
            selected_features = self._select_features(df, target_col='default')
            df = df[selected_features + ['default']]
            self.feature_names = selected_features
        else:
            # 在生产环境中，使用训练时选择的特征
            if self.feature_names:
                available_features = [f for f in self.feature_names if f in df.columns]
                df = df[available_features]

        logger.info(f"特征工程完成，特征数量: {len(df.columns)}")
        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建衍生特征"""
        # 数值特征交互
        if 'monthly_income' in df.columns and 'loan_amount' in df.columns:
            df['loan_to_income_ratio'] = df['loan_amount'] / (df['monthly_income'] + 1e-6)

        if 'age' in df.columns and 'employment_years' in df.columns:
            df['age_employment_ratio'] = df['employment_years'] / (df['age'] - 18 + 1e-6)

        # 分箱特征
        for col in ['age', 'monthly_income', 'credit_score']:
            if col in df.columns:
                df[f'{col}_bin'] = pd.qcut(df[col], q=5, duplicates='drop', labels=False)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        from sklearn.impute import SimpleImputer, KNNImputer

        imputation_method = self.config.get('features.imputation', 'knn')

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if imputation_method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        elif imputation_method == 'median':
            imputer = SimpleImputer(strategy='median')
        else:  # mean
            imputer = SimpleImputer(strategy='mean')

        if numeric_cols:
            df_numeric = df[numeric_cols]
            df_numeric_imputed = imputer.fit_transform(df_numeric)
            df[numeric_cols] = df_numeric_imputed
            self.imputer = imputer

        # 类别特征用众数
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'missing')

        return df

    def _select_features(self, df: pd.DataFrame, target_col: str = 'default') -> List[str]:
        """特征选择"""
        from sklearn.feature_selection import SelectKBest, f_classif

        X = df.drop(target_col, axis=1, errors='ignore')
        y = df[target_col]

        # 移除ID列和时间列
        cols_to_remove = ['application_id', 'application_date', 'default_probability']
        X = X[[col for col in X.columns if col not in cols_to_remove]]

        # 使用IV值选择
        if self.woe_info:
            iv_scores = {col: info.get('iv', 0) for col, info in self.woe_info.items()}
            # 选择IV > 0.02的特征
            selected = [col for col, iv in iv_scores.items() if iv > 0.02]

            # 确保有足够特征
            if len(selected) < 5:
                # 使用统计检验补充
                selector = SelectKBest(f_classif, k=10)
                selector.fit(X, y)
                selected = X.columns[selector.get_support()].tolist()[:10]
        else:
            # 使用统计检验
            selector = SelectKBest(f_classif, k=10)
            selector.fit(X, y)
            selected = X.columns[selector.get_support()].tolist()

        logger.info(f"特征选择完成，选择 {len(selected)} 个特征")
        return selected

    def get_feature_info(self) -> Dict:
        """获取特征信息"""
        return {
            'feature_names': self.feature_names,
            'woe_info': self.woe_info,
            'num_features': len(self.feature_names)
        }


# ============================================================================
# 模块五：WOE编码器
# ============================================================================

class WOEEncoder:
    """WOE编码器 - 证据权重编码"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.woe_info = {}
        self.bin_mappings = {}

    def fit_transform(self, data: pd.DataFrame, target_col: str = 'default') -> pd.DataFrame:
        """拟合并转换数据"""
        df = data.copy()
        X = df.drop(target_col, axis=1, errors='ignore')
        y = df[target_col]

        result_df = pd.DataFrame(index=df.index)

        for col in tqdm(X.columns, desc="WOE编码"):
            if col in ['application_id', 'application_date']:
                result_df[col] = df[col]
                continue

            if X[col].nunique() > 20:  # 数值特征
                iv, bin_stats = self._calculate_woe_iv_numeric(X[col], y)
                if iv > self.config.get('features.woe.iv_threshold', 0.02):
                    # 创建WOE编码
                    woe_mapping = self._create_woe_mapping(bin_stats)
                    result_df[f'{col}_woe'] = X[col].apply(
                        lambda x: self._map_to_woe_numeric(x, woe_mapping)
                    )
                    self.woe_info[col] = {
                        'iv': iv,
                        'bins': woe_mapping,
                        'type': 'numeric'
                    }
                    self.bin_mappings[col] = woe_mapping
            else:  # 类别特征
                woe_mapping = self._calculate_woe_categorical(X[col], y)
                iv = self._calculate_iv_from_mapping(woe_mapping, X[col], y)
                if iv > self.config.get('features.woe.iv_threshold', 0.02):
                    result_df[f'{col}_woe'] = X[col].map(woe_mapping)
                    self.woe_info[col] = {
                        'iv': iv,
                        'bins': woe_mapping,
                        'type': 'categorical'
                    }
                    self.bin_mappings[col] = woe_mapping

        # 添加目标变量
        if target_col in df.columns:
            result_df[target_col] = y

        return result_df

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换新数据"""
        df = data.copy()
        result_df = pd.DataFrame(index=df.index)

        for col, info in self.bin_mappings.items():
            if col in df.columns:
                if info['type'] == 'numeric':
                    result_df[f'{col}_woe'] = df[col].apply(
                        lambda x: self._map_to_woe_numeric(x, info['bins'])
                    )
                else:  # categorical
                    result_df[f'{col}_woe'] = df[col].map(info['bins'])

        # 保留原始列（如果没有WOE编码）
        for col in df.columns:
            if f'{col}_woe' not in result_df.columns and col not in self.bin_mappings:
                result_df[col] = df[col]

        return result_df

    def _calculate_woe_iv_numeric(self, feature: pd.Series, target: pd.Series, n_bins: int = 10):
        """计算数值特征的WOE和IV"""
        # 等频分箱
        try:
            bins = pd.qcut(feature, q=n_bins, duplicates='drop')
        except:
            # 如果不能分箱，返回0
            return 0, pd.DataFrame()

        bin_stats = []
        for bin_range in bins.cat.categories:
            mask = (feature >= bin_range.left) & (feature < bin_range.right)
            if mask.sum() > 0:
                good = target[mask].eq(0).sum()
                bad = target[mask].eq(1).sum()
                total_good = target.eq(0).sum()
                total_bad = target.eq(1).sum()

                dist_good = good / (total_good + 1e-6)
                dist_bad = bad / (total_bad + 1e-6)
                woe = np.log((dist_good + 1e-6) / (dist_bad + 1e-6))
                iv = (dist_good - dist_bad) * woe

                bin_stats.append({
                    'bin': bin_range,
                    'woe': woe,
                    'iv': iv,
                    'good': good,
                    'bad': bad
                })

        iv_total = sum([b['iv'] for b in bin_stats])
        return iv_total, pd.DataFrame(bin_stats)

    def _create_woe_mapping(self, bin_stats: pd.DataFrame) -> Dict:
        """创建WOE映射"""
        mapping = {}
        for _, row in bin_stats.iterrows():
            mapping[row['bin']] = row['woe']
        return mapping

    def _map_to_woe_numeric(self, x: float, woe_mapping: Dict) -> float:
        """将数值映射到WOE值"""
        for bin_range, woe in woe_mapping.items():
            if bin_range.left <= x < bin_range.right:
                return woe
        return 0  # 默认值

    def _calculate_woe_categorical(self, feature: pd.Series, target: pd.Series) -> Dict:
        """计算类别特征的WOE"""
        woe_mapping = {}

        for val in feature.dropna().unique():
            mask = feature == val
            if mask.sum() > 0:
                good = target[mask].eq(0).sum()
                bad = target[mask].eq(1).sum()
                total_good = target.eq(0).sum()
                total_bad = target.eq(1).sum()

                dist_good = good / (total_good + 1e-6)
                dist_bad = bad / (total_bad + 1e-6)
                woe = np.log((dist_good + 1e-6) / (dist_bad + 1e-6))
                woe_mapping[val] = woe

        return woe_mapping

    def _calculate_iv_from_mapping(self, woe_mapping: Dict, feature: pd.Series, target: pd.Series) -> float:
        """从WOE映射计算IV"""
        iv_total = 0
        total_good = target.eq(0).sum()
        total_bad = target.eq(1).sum()

        for val, woe in woe_mapping.items():
            mask = feature == val
            if mask.sum() > 0:
                good = target[mask].eq(0).sum()
                bad = target[mask].eq(1).sum()
                dist_good = good / (total_good + 1e-6)
                dist_bad = bad / (total_bad + 1e-6)
                iv_total += (dist_good - dist_bad) * woe

        return iv_total


# ============================================================================
# 模块六：模型训练器
# ============================================================================

class ModelTrainer:
    """模型训练器 - 训练、评估、保存模型"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.evaluator = ModelEvaluator(config)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict:
        """训练模型"""
        logger.info("开始模型训练...")

        # 准备数据
        X_train_processed, X_test_processed = self._prepare_data(X_train, X_test)

        # 获取模型配置
        algorithms = self.config.get('model.algorithms', ['logistic', 'random_forest', 'lightgbm'])

        # 训练基础模型
        for algo in algorithms:
            try:
                model = self._train_single_model(algo, X_train_processed, y_train)
                if model is not None:
                    self.models[algo] = model
                    logger.info(f"模型 {algo} 训练完成")
            except Exception as e:
                logger.error(f"模型 {algo} 训练失败: {e}")

        # 训练集成模型
        if len(self.models) >= 2:
            ensemble_model = self._train_ensemble_model(X_train_processed, y_train)
            if ensemble_model is not None:
                self.models['ensemble'] = ensemble_model

        # 评估模型
        if X_test is not None and y_test is not None:
            evaluation_results = self.evaluator.evaluate_all(
                self.models, X_test_processed, y_test
            )

            # 选择最佳模型
            self.best_model_name, self.best_model = self._select_best_model(evaluation_results)

            logger.info(f"最佳模型: {self.best_model_name}")

            return {
                'models': self.models,
                'best_model': self.best_model,
                'best_model_name': self.best_model_name,
                'evaluation_results': evaluation_results,
                'scaler': self.scaler
            }

        return {'models': self.models, 'scaler': self.scaler}

    def _prepare_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None):
        """准备数据"""
        from sklearn.preprocessing import StandardScaler

        # 数值特征标准化
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        self.scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])

        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            return X_train_scaled, X_test_scaled

        return X_train_scaled, None

    def _train_single_model(self, algorithm: str, X: pd.DataFrame, y: pd.Series):
        """训练单个模型"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        import lightgbm as lgb
        import xgboost as xgb

        if algorithm == 'logistic':
            model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
        elif algorithm == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == 'lightgbm':
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            )
        elif algorithm == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        else:
            logger.warning(f"未知算法: {algorithm}")
            return None

        model.fit(X, y)
        return model

    def _train_ensemble_model(self, X: pd.DataFrame, y: pd.Series):
        """训练集成模型"""
        from sklearn.ensemble import VotingClassifier

        if len(self.models) < 2:
            return None

        # 选择前3个模型进行集成
        top_models = list(self.models.items())[:3]
        estimators = [(name, model) for name, model in top_models]

        ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )

        ensemble_model.fit(X, y)
        return ensemble_model

    def _select_best_model(self, evaluation_results: Dict) -> tuple:
        """选择最佳模型"""
        best_name = None
        best_auc = 0

        for name, results in evaluation_results.items():
            if 'test_auc' in results and results['test_auc'] > best_auc:
                best_auc = results['test_auc']
                best_name = name

        if best_name is not None:
            return best_name, self.models[best_name]

        # 默认返回第一个模型
        first_name = list(self.models.keys())[0]
        return first_name, self.models[first_name]

    def save_models(self, output_dir: str = 'models'):
        """保存模型"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 保存所有模型
        for name, model in self.models.items():
            model_path = Path(output_dir) / f'model_{name}.pkl'
            joblib.dump(model, model_path)
            logger.info(f"模型 {name} 保存到 {model_path}")

        # 保存最佳模型
        if self.best_model is not None:
            best_model_path = Path(output_dir) / 'best_model.pkl'
            joblib.dump(self.best_model, best_model_path)
            logger.info(f"最佳模型保存到 {best_model_path}")

        # 保存scaler
        if self.scaler is not None:
            scaler_path = Path(output_dir) / 'scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler保存到 {scaler_path}")

        # 保存模型信息
        model_info = {
            'best_model_name': self.best_model_name,
            'models_trained': list(self.models.keys()),
            'timestamp': datetime.now().isoformat()
        }

        info_path = Path(output_dir) / 'model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        return output_dir


# ============================================================================
# 模块七：模型评估器
# ============================================================================

class ModelEvaluator:
    """模型评估器 - 评估模型性能"""

    def __init__(self, config: ConfigManager):
        self.config = config

    def evaluate_all(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """评估所有模型"""
        results = {}

        for name, model in models.items():
            try:
                results[name] = self.evaluate_single(model, X_test, y_test)
            except Exception as e:
                logger.error(f"模型 {name} 评估失败: {e}")
                results[name] = {'error': str(e)}

        return results

    def evaluate_single(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """评估单个模型"""
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, precision_score,
            recall_score, f1_score, confusion_matrix, classification_report
        )

        # 预测
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = None

        # 计算指标
        results = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_pred, zero_division=0),
        }

        if y_pred_proba is not None:
            results['test_auc'] = roc_auc_score(y_test, y_pred_proba)

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm.tolist()

        # 分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        results['classification_report'] = report

        return results

    def generate_evaluation_report(self, results: Dict, output_path: str = 'reports/evaluation_report.html'):
        """生成评估报告"""
        import pandas as pd

        # 创建汇总表
        summary_data = []
        for model_name, metrics in results.items():
            if 'error' in metrics:
                continue

            summary_data.append({
                'Model': model_name,
                'AUC': metrics.get('test_auc', 0),
                'Accuracy': metrics.get('test_accuracy', 0),
                'Precision': metrics.get('test_precision', 0),
                'Recall': metrics.get('test_recall', 0),
                'F1': metrics.get('test_f1', 0)
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('AUC', ascending=False)

        # 保存报告
        Path('reports').mkdir(parents=True, exist_ok=True)

        # 简单文本报告
        report_text = f"""
        模型评估报告
        生成时间: {datetime.now().isoformat()}

        模型性能排名:
        {summary_df.to_string(index=False)}

        最佳模型: {summary_df.iloc[0]['Model']}
        最佳AUC: {summary_df.iloc[0]['AUC']:.4f}
        """

        with open(output_path.replace('.html', '.txt'), 'w') as f:
            f.write(report_text)

        logger.info(f"评估报告保存到 {output_path}")
        return summary_df


# ============================================================================
# 模块八：评分卡转换器
# ============================================================================

class ScorecardTransformer:
    """评分卡转换器 - 将模型转换为评分卡"""

    def __init__(self, config: ConfigManager):
        self.config = config

    def transform(self, model, feature_names: List[str],
                  base_score: int = None, pdo: int = None, odds: int = None) -> Dict:
        """将逻辑回归模型转换为评分卡"""
        if not hasattr(model, 'coef_'):
            raise ValueError("模型必须为逻辑回归模型")

        if base_score is None:
            base_score = self.config.get('scoring.base_score', 600)
        if pdo is None:
            pdo = self.config.get('scoring.pdo', 20)
        if odds is None:
            odds = self.config.get('scoring.odds', 20)

        # 评分卡参数
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(odds)

        # 获取系数
        intercept = model.intercept_[0]
        coefficients = model.coef_[0]

        # 创建评分卡
        scorecard = pd.DataFrame({
            'feature': ['Intercept'] + feature_names,
            'coefficient': [intercept] + list(coefficients)
        })

        # 计算分数
        scorecard['points'] = -factor * scorecard['coefficient']

        # 计算基础分
        base_points = offset - factor * intercept

        return {
            'scorecard': scorecard,
            'parameters': {
                'base_score': base_score,
                'pdo': pdo,
                'odds': odds,
                'factor': factor,
                'offset': offset,
                'base_points': base_points
            },
            'feature_points': dict(zip(feature_names, -factor * coefficients))
        }

    def calculate_score(self, features: Dict, scorecard_data: Dict) -> float:
        """计算信用分数"""
        total_score = scorecard_data['parameters']['base_points']

        for feature, value in features.items():
            if feature in scorecard_data['feature_points']:
                total_score += scorecard_data['feature_points'][feature] * value

        return total_score

    def save_scorecard(self, scorecard_data: Dict, output_path: str = 'models/scorecard.json'):
        """保存评分卡"""
        # 转换为可序列化格式
        serializable_data = {
            'scorecard': scorecard_data['scorecard'].to_dict('records'),
            'parameters': scorecard_data['parameters'],
            'feature_points': scorecard_data['feature_points']
        }

        Path('models').mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        logger.info(f"评分卡保存到 {output_path}")
        return output_path


# ============================================================================
# 模块九：模型监控器
# ============================================================================

class ModelMonitor:
    """模型监控器 - 监控模型性能和稳定性"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.monitoring_data = []

    def calculate_psi(self, expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
        """计算PSI（群体稳定性指标）"""
        # 分箱
        expected_bins = pd.qcut(expected, q=bins, duplicates='drop')
        actual_bins = pd.qcut(actual, q=bins, labels=expected_bins.cat.categories)

        # 计算分布
        expected_dist = expected_bins.value_counts(normalize=True).sort_index()
        actual_dist = actual_bins.value_counts(normalize=True).sort_index()

        # 计算PSI
        psi = 0
        for bin_val in expected_dist.index:
            exp_pct = expected_dist[bin_val]
            act_pct = actual_dist.get(bin_val, 0)
            if exp_pct > 0 and act_pct > 0:
                psi += (act_pct - exp_pct) * np.log(act_pct / exp_pct)

        return psi

    def check_model_decay(self, current_performance: Dict, baseline_performance: Dict) -> Dict:
        """检查模型衰减"""
        decay_report = {}

        for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
            if metric in current_performance and metric in baseline_performance:
                current_val = current_performance[metric]
                baseline_val = baseline_performance[metric]
                decay_pct = (baseline_val - current_val) / baseline_val

                decay_report[metric] = {
                    'current': current_val,
                    'baseline': baseline_val,
                    'decay_pct': decay_pct,
                    'alert': decay_pct > self.config.get('monitoring.auc_decay_threshold', 0.05)
                }

        return decay_report

    def monitor_feature_stability(self, train_features: pd.DataFrame,
                                  current_features: pd.DataFrame) -> Dict:
        """监控特征稳定性"""
        stability_report = {}

        for col in train_features.columns:
            if col in current_features.columns:
                psi = self.calculate_psi(
                    train_features[col],
                    current_features[col]
                )

                stability_report[col] = {
                    'psi': psi,
                    'alert': psi > self.config.get('monitoring.psi_threshold', 0.1)
                }

        return stability_report

    def generate_monitoring_report(self, monitoring_data: List[Dict],
                                   output_path: str = 'reports/monitoring_report.json'):
        """生成监控报告"""
        report = {
            'generated_time': datetime.now().isoformat(),
            'monitoring_data': monitoring_data,
            'summary': {
                'total_checks': len(monitoring_data),
                'alerts': sum(1 for d in monitoring_data if d.get('has_alert', False))
            }
        }

        Path('reports').mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"监控报告保存到 {output_path}")
        return report


# ============================================================================
# 模块十：API服务
# ============================================================================

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import werkzeug


class CreditScoringAPI:
    """信用评分API服务"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.app = Flask(__name__)
        self.api = Api(self.app,
                       version='1.0',
                       title='信用评分API',
                       description='提供实时信用评分服务')

        # 加载模型
        self.model = None
        self.scaler = None
        self.scorecard = None
        self.feature_engineer = None

        self.load_models()
        self.setup_routes()

    def load_models(self):
        """加载模型"""
        try:
            # 加载最佳模型
            self.model = joblib.load('models/best_model.pkl')
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None

        try:
            # 加载scaler
            self.scaler = joblib.load('models/scaler.pkl')
            logger.info("Scaler加载成功")
        except Exception as e:
            logger.error(f"Scaler加载失败: {e}")
            self.scaler = None

        try:
            # 加载评分卡
            with open('models/scorecard.json', 'r') as f:
                self.scorecard = json.load(f)
            logger.info("评分卡加载成功")
        except Exception as e:
            logger.error(f"评分卡加载失败: {e}")
            self.scorecard = None

        try:
            # 加载特征工程
            self.feature_engineer = joblib.load('models/pipeline.pkl')
            logger.info("特征工程管道加载成功")
        except Exception as e:
            logger.error(f"特征工程管道加载失败: {e}")
            self.feature_engineer = None

    def setup_routes(self):
        """设置API路由"""

        # 请求模型
        scoring_request = self.api.model('ScoringRequest', {
            'application_id': fields.String(required=True, description='申请ID'),
            'age': fields.Float(required=True, description='年龄'),
            'monthly_income': fields.Float(required=True, description='月收入'),
            'employment_years': fields.Float(required=True, description='工作年限'),
            'credit_score': fields.Float(required=True, description='信用分数'),
            'loan_amount': fields.Float(required=True, description='贷款金额'),
            'debt_to_income': fields.Float(required=True, description='负债收入比'),
            'num_delinquencies': fields.Float(required=True, description='违约记录数'),
            'num_credit_inquiries': fields.Float(required=True, description='信用查询次数')
        })

        scoring_response = self.api.model('ScoringResponse', {
            'application_id': fields.String(description='申请ID'),
            'score': fields.Float(description='信用分数'),
            'default_probability': fields.Float(description='违约概率'),
            'risk_level': fields.String(description='风险等级'),
            'decision': fields.String(description='审批决定'),
            'features': fields.Raw(description='特征值'),
            'model_version': fields.String(description='模型版本'),
            'timestamp': fields.String(description='时间戳')
        })

        batch_scoring_request = self.api.model('BatchScoringRequest', {
            'applications': fields.List(fields.Nested(scoring_request), required=True, description='申请列表')
        })

        batch_scoring_response = self.api.model('BatchScoringResponse', {
            'results': fields.List(fields.Nested(scoring_response), description='评分结果列表'),
            'total_processed': fields.Integer(description='处理总数'),
            'success_count': fields.Integer(description='成功数量'),
            'failed_count': fields.Integer(description='失败数量'),
            'processing_time': fields.Float(description='处理时间(秒)')
        })

        # 单条评分端点
        @self.api.route('/score')
        class SingleScoring(Resource):
            @self.api.expect(scoring_request)
            @self.api.marshal_with(scoring_response)
            def post(self):
                """单条信用评分"""
                start_time = datetime.now()

                try:
                    data = request.json
                    result = self.process_single_application(data)

                    result['processing_time'] = (datetime.now() - start_time).total_seconds()
                    result['model_version'] = '1.0.0'
                    result['timestamp'] = datetime.now().isoformat()

                    logger.info(f"单条评分完成: {data.get('application_id')}")
                    return result, 200

                except Exception as e:
                    logger.error(f"单条评分失败: {e}")
                    return {
                        'error': str(e),
                        'application_id': data.get('application_id', 'unknown')
                    }, 400

        # 批量评分端点
        @self.api.route('/batch_score')
        class BatchScoring(Resource):
            @self.api.expect(batch_scoring_request)
            @self.api.marshal_with(batch_scoring_response)
            def post(self):
                """批量信用评分"""
                start_time = datetime.now()

                try:
                    data = request.json
                    applications = data.get('applications', [])

                    results = []
                    success_count = 0
                    failed_count = 0

                    for app in applications:
                        try:
                            result = self.process_single_application(app)
                            results.append(result)
                            success_count += 1
                        except Exception as e:
                            logger.error(f"批量评分失败: {e}")
                            failed_count += 1

                    processing_time = (datetime.now() - start_time).total_seconds()

                    logger.info(f"批量评分完成: 成功 {success_count}, 失败 {failed_count}")

                    return {
                        'results': results,
                        'total_processed': len(applications),
                        'success_count': success_count,
                        'failed_count': failed_count,
                        'processing_time': processing_time
                    }, 200

                except Exception as e:
                    logger.error(f"批量评分处理失败: {e}")
                    return {'error': str(e)}, 400

        # 健康检查端点
        @self.api.route('/health')
        class HealthCheck(Resource):
            def get(self):
                """健康检查"""
                return {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'model_loaded': self.model is not None,
                    'scaler_loaded': self.scaler is not None,
                    'scorecard_loaded': self.scorecard is not None
                }, 200

        # 模型信息端点
        @self.api.route('/model_info')
        class ModelInfo(Resource):
            def get(self):
                """获取模型信息"""
                try:
                    with open('models/model_info.json', 'r') as f:
                        model_info = json.load(f)

                    return {
                        'model_info': model_info,
                        'config': self.config.config
                    }, 200
                except Exception as e:
                    logger.error(f"获取模型信息失败: {e}")
                    return {'error': str(e)}, 500

    def process_single_application(self, application_data: Dict) -> Dict:
        """处理单个申请"""
        # 1. 提取特征
        features = self.extract_features(application_data)

        # 2. 特征工程
        processed_features = self.process_features(features)

        # 3. 预测
        if self.model and hasattr(self.model, 'predict_proba'):
            # 使用机器学习模型
            default_prob = self.model.predict_proba([processed_features])[0, 1]

            # 计算信用分数（基于概率）
            score = self.calculate_score_from_probability(default_prob)
        elif self.scorecard:
            # 使用评分卡
            score = self.calculate_score_from_scorecard(features)
            default_prob = self.calculate_probability_from_score(score)
        else:
            raise ValueError("没有可用的模型或评分卡")

        # 4. 风险等级
        risk_level = self.get_risk_level(score)

        # 5. 审批决定
        decision = self.get_decision(score, risk_level)

        return {
            'application_id': application_data.get('application_id'),
            'score': float(score),
            'default_probability': float(default_prob),
            'risk_level': risk_level,
            'decision': decision,
            'features': features
        }

    def extract_features(self, application_data: Dict) -> Dict:
        """提取特征"""
        # 这里可以根据需要添加更多的特征提取逻辑
        return {
            'age': application_data.get('age'),
            'monthly_income': application_data.get('monthly_income'),
            'employment_years': application_data.get('employment_years'),
            'credit_score': application_data.get('credit_score'),
            'loan_amount': application_data.get('loan_amount'),
            'debt_to_income': application_data.get('debt_to_income'),
            'num_delinquencies': application_data.get('num_delinquencies'),
            'num_credit_inquiries': application_data.get('num_credit_inquiries')
        }

    def process_features(self, features: Dict) -> List:
        """处理特征"""
        # 转换为DataFrame
        df = pd.DataFrame([features])

        # 应用特征工程
        if self.feature_engineer:
            processed_df = self.feature_engineer.transform(df)
        else:
            processed_df = df

        # 应用scaler
        if self.scaler:
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            processed_df[numeric_cols] = self.scaler.transform(processed_df[numeric_cols])

        # 转换为列表
        return processed_df.values.tolist()[0]

    def calculate_score_from_probability(self, probability: float) -> float:
        """从概率计算信用分数"""
        # 简化计算，实际应该基于评分卡
        # 概率越低，分数越高
        base_score = 600
        pdo = 20

        # 确保概率在合理范围内
        probability = max(0.001, min(0.999, probability))

        # 计算odds
        odds = (1 - probability) / probability

        # 计算分数
        score = base_score + (pdo / np.log(2)) * np.log(odds)

        return float(max(300, min(850, score)))

    def calculate_score_from_scorecard(self, features: Dict) -> float:
        """从评分卡计算分数"""
        if not self.scorecard:
            return 600.0

        total_score = self.scorecard['parameters']['base_points']

        for feature, value in features.items():
            if feature in self.scorecard['feature_points']:
                total_score += self.scorecard['feature_points'][feature] * value

        return float(max(300, min(850, total_score)))

    def calculate_probability_from_score(self, score: float) -> float:
        """从分数计算违约概率"""
        base_score = 600
        pdo = 20

        # 计算odds
        odds = np.exp((score - base_score) * np.log(2) / pdo)

        # 计算概率
        probability = 1 / (1 + odds)

        return float(probability)

    def get_risk_level(self, score: float) -> str:
        """获取风险等级"""
        if score >= 800:
            return '优秀'
        elif score >= 740:
            return '很好'
        elif score >= 670:
            return '良好'
        elif score >= 580:
            return '一般'
        else:
            return '较差'

    def get_decision(self, score: float, risk_level: str) -> str:
        """获取审批决定"""
        if score >= 670:
            return '批准'
        elif score >= 600:
            return '有条件批准'
        else:
            return '拒绝'

    def run(self, host: str = None, port: int = None, debug: bool = None):
        """运行API服务"""
        if host is None:
            host = self.config.get('api.host', '0.0.0.0')
        if port is None:
            port = self.config.get('api.port', 5000)
        if debug is None:
            debug = self.config.get('api.debug', False)

        logger.info(f"启动API服务: http://{host}:{port}")
        logger.info(f"API文档: http://{host}:{port}/")

        self.app.run(host=host, port=port, debug=debug)


# ============================================================================
# 模块十一：主程序
# ============================================================================

def main():
    """主程序"""
    # 1. 加载配置
    config = ConfigManager()

    # 2. 训练模式
    if config.get('mode', 'train') == 'train':
        logger.info("开始训练模式...")

        # 2.1 数据管道
        pipeline = DataPipeline(config)
        data = pipeline.run(mode='train')

        # 2.2 模型训练
        trainer = ModelTrainer(config)
        training_results = trainer.train(
            data['X_train'], data['y_train'],
            data['X_test'], data['y_test']
        )

        # 2.3 评估
        evaluator = ModelEvaluator(config)
        evaluation_report = evaluator.generate_evaluation_report(
            training_results['evaluation_results']
        )

        # 2.4 评分卡转换
        if 'logistic' in training_results['models']:
            transformer = ScorecardTransformer(config)
            logistic_model = training_results['models']['logistic']

            # 获取特征名称
            feature_names = data['X_train'].columns.tolist()

            # 转换
            scorecard_data = transformer.transform(
                logistic_model,
                feature_names
            )

            # 保存评分卡
            transformer.save_scorecard(scorecard_data)

        # 2.5 保存模型
        trainer.save_models()

        # 2.6 保存数据管道
        pipeline.save_pipeline()

        logger.info("训练完成！")

    # 3. API服务模式
    else:
        logger.info("启动API服务模式...")

        # 3.1 创建API服务
        api_service = CreditScoringAPI(config)

        # 3.2 运行API
        api_service.run(
            host=config.get('api.host'),
            port=config.get('api.port'),
            debug=config.get('api.debug')
        )


if __name__ == "__main__":
    main()