library(quantmod)
library(ggplot2)
library(zoo)

# 下载数据
symbols <- c("AMZN", "AAPL", "MSFT", "GOOG")
getSymbols(symbols, src = "yahoo", from = "2019-01-01", to = "2024-12-31")

# 提取收盘价
prices <- do.call(merge, lapply(symbols, function(sym) Cl(get(sym))))

# 计算每日收益率并处理缺失值
returns <- na.omit(ROC(prices, type = "discrete"))

# 计算12个月动量因子（即252个交易日的累计收益率）
momentum <- lag(prices, 252) / prices - 1
momentum <- na.omit(momentum)

# 确保收益率和动量因子的行数匹配
common_dates <- intersect(index(returns), index(momentum))
matched_returns <- returns[common_dates]
matched_momentum <- momentum[common_dates]

# 设置投资组合的权重（等权重）
weights <- rep(1/length(symbols), length(symbols))
names(weights) <- symbols

# 计算组合收益率
portfolio_returns <- rowSums(matched_returns * weights)

# VaR计算函数
historical_VaR <- function(returns, confidence_level = 0.95) {
  -quantile(returns, 1 - confidence_level)
}

normal_VaR <- function(returns, confidence_level = 0.95) {
  mean(returns) - qnorm(1 - confidence_level) * sd(returns)
}

# 计算VaR
VaR_historical <- historical_VaR(portfolio_returns)
VaR_normal <- normal_VaR(portfolio_returns)

cat("Historical VaR:", VaR_historical, "\n")
cat("Normal VaR:", VaR_normal, "\n")

# 构建因子回归模型（这里我们使用动量因子）
factor_model <- lm(portfolio_returns ~ matched_momentum)

# 提取模型结果
summary(factor_model)

# 计算总风险和跟踪误差
predicted_returns <- predict(factor_model)
tracking_error <- sd(predicted_returns - mean(portfolio_returns))
total_risk <- sd(portfolio_returns)

cat("Total Risk:", total_risk, "\n")
cat("Tracking Error:", tracking_error, "\n")

# 风险分解到个别因子
individual_factor_contribution <- coef(factor_model)[-1] * sd(matched_momentum)
names(individual_factor_contribution) <- "Momentum Factor Contribution"

# 打印个别因子风险贡献
cat("Individual Factor Contribution to Total Risk:\n")
print(individual_factor_contribution)

# 计算动量因子在总风险中的占比
total_risk_contribution <- sum(individual_factor_contribution^2)
percentage_factor_contribution <- (individual_factor_contribution^2) / total_risk_contribution
cat("Percentage Contribution of Momentum Factor to Total Risk:\n")
print(percentage_factor_contribution)

# 风险分解
systematic_risk <- sqrt(sum((coef(factor_model)[-1] * sd(matched_momentum))^2))
idiosyncratic_risk <- sd(residuals(factor_model))

cat("Systematic Risk:", systematic_risk, "\n")
cat("Idiosyncratic Risk:", idiosyncratic_risk, "\n")

# 主要风险贡献者
individual_contributions <- coef(factor_model)[-1] * sd(matched_momentum)
names(individual_contributions) <- paste("Contribution of", symbols)
cat("Individual Risk Contributions:\n")
print(individual_contributions)

# 计算每个股票的风险贡献占总风险的比例
total_contribution <- sum(individual_contributions^2)
percentage_contributions <- (individual_contributions^2) / total_contribution
cat("Percentage Contributions:\n")
print(percentage_contributions)

confidence_level <- 0.95

# 计算历史VaR和正态VaR随时间变化
VaR_values <- data.frame(
  date = index(returns),
  Historical_VaR = rollapply(returns, width = 90, FUN = historical_VaR, fill = NA, align = "right", by.column = FALSE, confidence_level),
  Normal_VaR = rollapply(returns, width = 90, FUN = normal_VaR, fill = NA, align = "right", by.column = FALSE, confidence_level)
)

# 绘图
ggplot(VaR_values, aes(x = date)) +
  geom_line(aes(y = Historical_VaR, color = "Historical VaR")) +
  geom_line(aes(y = Normal_VaR, color = "Normal VaR")) +
  labs(title = "VaR Over Time", x = "Date", y = "VaR", color = "VaR Type") +
  theme_minimal()