# 导入必要的库
library(quantmod)
library(PerformanceAnalytics)
library(quadprog)
library(ggplot2)
library(reshape2)

# 获取数据
symbols <- c("GOOGL", "AMZN", "AAPL", "MSFT")
getSymbols(symbols, from = "2019-01-01", to = "2024-12-31")
prices <- na.omit(merge(Cl(GOOGL), Cl(AMZN), Cl(AAPL), Cl(MSFT)))
returns <- na.omit(Return.calculate(prices))

# 设置再平衡频率
rebalance_freq <- "months" 

# 定义原始权重
original_weights <- rep(1 / length(symbols), length(symbols))

# 定义计算函数
min_TE_portfolio <- function(original_weights, cov_matrix) {
  Dmat <- cov_matrix
  dvec <- rep(0, ncol(cov_matrix))
  Amat <- cbind(1, diag(ncol(cov_matrix)))
  bvec <- c(1, rep(0, ncol(cov_matrix)))
  result <- solve.QP(Dmat, dvec, Amat, bvec, meq=1)
  return(result$solution)
}

mvo_portfolio <- function(cov_matrix) {
  mean_returns <- colMeans(returns)
  Dmat <- cov_matrix
  dvec <- mean_returns
  Amat <- cbind(1, diag(ncol(cov_matrix)))
  bvec <- c(1, rep(0, ncol(cov_matrix)))
  result <- solve.QP(Dmat, dvec, Amat, bvec, meq=1)
  return(result$solution)
}

# 历史VaR计算函数
calculate_var <- function(returns, weights, p = 0.95) {
  portfolio_returns <- rowSums(returns * weights)
  VaR <- -quantile(portfolio_returns, probs = 1 - p, na.rm = TRUE)
  return(VaR)
}

# 正态分布VaR计算函数
calculate_var_norm <- function(returns, weights, p = 0.95) {
  portfolio_returns <- rowSums(returns * weights)
  mean_return <- mean(portfolio_returns)
  sd_return <- sd(portfolio_returns)
  z_score <- qnorm(1-p)
  VaR_norm <- mean_return - z_score * sd_return
  return(VaR_norm)
}

# 再平衡策略
rebalance_dates <- endpoints(returns, on = rebalance_freq)
performance_summary <- data.frame(Date = index(returns)[rebalance_dates[-1]])

# 为各个组合储存表现与风险数据
performance_summary$Original_Return <- 0
performance_summary$Min_TE_Return <- 0
performance_summary$MVO_Return <- 0
performance_summary$Original_Volatility <- 0
performance_summary$Min_TE_Volatility <- 0
performance_summary$MVO_Volatility <- 0
performance_summary$Original_Sharpe <- 0
performance_summary$Min_TE_Sharpe <- 0
performance_summary$MVO_Sharpe <- 0
performance_summary$Total_Risk_Original <- 0
performance_summary$Total_Risk_Min_TE <- 0
performance_summary$Total_Risk_MVO <- 0
performance_summary$Tracking_Error_Min_TE <- 0
performance_summary$Tracking_Error_MVO <- 0
performance_summary$Tracking_Error_Original <- 0  
performance_summary$VaR_Historical_Original <- 0
performance_summary$VaR_Historical_Min_TE <- 0
performance_summary$VaR_Historical_MVO <- 0
performance_summary$VaR_Normal_Original <- 0
performance_summary$VaR_Normal_Min_TE <- 0
performance_summary$VaR_Normal_MVO <- 0

# Key: Initialize Risk Contribution
performance_summary[, paste0("Risk_Contribution_", symbols)] <- 0
performance_summary[, paste0("Total_Risk_Contribution_", symbols)] <- 0

for (i in 1:(length(rebalance_dates) - 1)) {
  current_returns <- returns[(rebalance_dates[i] + 1):rebalance_dates[i + 1], ]
  cov_matrix <- cov(current_returns)
  
  # 计算组合权重
  min_te_weights <- min_TE_portfolio(original_weights, cov_matrix)
  mvo_weights <- mvo_portfolio(cov_matrix)
  
  # 计算组合收益
  original_returns <- rowSums(current_returns * original_weights)
  min_te_returns <- rowSums(current_returns * min_te_weights)
  mvo_returns <- rowSums(current_returns * mvo_weights)
  
  # 年化回报率
  performance_summary$Original_Return[i] <- mean(original_returns) * 252
  performance_summary$Min_TE_Return[i] <- mean(min_te_returns) * 252
  performance_summary$MVO_Return[i] <- mean(mvo_returns) * 252
  
  # 年化波动率
  performance_summary$Original_Volatility[i] <- sd(original_returns) * sqrt(252)
  performance_summary$Min_TE_Volatility[i] <- sd(min_te_returns) * sqrt(252)
  performance_summary$MVO_Volatility[i] <- sd(mvo_returns) * sqrt(252)
  
  # 年化夏普率
  performance_summary$Original_Sharpe[i] <- (mean(original_returns) * 252) / (sd(original_returns) * sqrt(252))
  performance_summary$Min_TE_Sharpe[i] <- (mean(min_te_returns) * 252) / (sd(min_te_returns) * sqrt(252))
  performance_summary$MVO_Sharpe[i] <- (mean(mvo_returns) * 252) / (sd(mvo_returns) * sqrt(252))
  
  # 风险和VaR分析
  performance_summary$Total_Risk_Original[i] <- sqrt(t(original_weights) %*% cov_matrix %*% original_weights)
  performance_summary$Total_Risk_Min_TE[i] <- sqrt(t(min_te_weights) %*% cov_matrix %*% min_te_weights)
  performance_summary$Total_Risk_MVO[i] <- sqrt(t(mvo_weights) %*% cov_matrix %*% mvo_weights)
  performance_summary$VaR_Historical_Original[i] <- calculate_var(current_returns, original_weights)
  performance_summary$VaR_Historical_Min_TE[i] <- calculate_var(current_returns, min_te_weights)
  performance_summary$VaR_Historical_MVO[i] <- calculate_var(current_returns, mvo_weights)
  performance_summary$VaR_Normal_Original[i] <- calculate_var_norm(current_returns, original_weights)
  performance_summary$VaR_Normal_Min_TE[i] <- calculate_var_norm(current_returns, min_te_weights)
  performance_summary$VaR_Normal_MVO[i] <- calculate_var_norm(current_returns, mvo_weights)
  
  # 计算跟踪误差
  performance_summary$Tracking_Error_Min_TE[i] <- sqrt(mean((min_te_returns - original_returns)^2))
  performance_summary$Tracking_Error_MVO[i] <- sqrt(mean((mvo_returns - original_returns)^2))
  performance_summary$Tracking_Error_Original[i] <- sqrt(mean((original_returns - original_returns)^2))
  
  # 风险分解 系统性和特定风险
  systematic_risk_min_te <- sum(min_te_weights * diag(cov_matrix) * min_te_weights)
  idiosyncratic_risk_min_te <- performance_summary$Total_Risk_Min_TE[i]^2 - systematic_risk_min_te
  performance_summary$Systematic_Risk_Min_TE[i] <- systematic_risk_min_te
  performance_summary$Idiosyncratic_Risk_Min_TE[i] <- idiosyncratic_risk_min_te
  
  systematic_risk_mvo <- sum(mvo_weights * diag(cov_matrix) * mvo_weights)
  idiosyncratic_risk_mvo <- performance_summary$Total_Risk_MVO[i]^2 - systematic_risk_mvo
  performance_summary$Systematic_Risk_MVO[i] <- systematic_risk_mvo
  performance_summary$Idiosyncratic_Risk_MVO[i] <- idiosyncratic_risk_mvo
  
  systematic_risk_original <- sum(original_weights * diag(cov_matrix) * original_weights)
  idiosyncratic_risk_original <- performance_summary$Total_Risk_Original[i]^2 - systematic_risk_original
  performance_summary$Systematic_Risk_Original[i] <- systematic_risk_original
  performance_summary$Idiosyncratic_Risk_Original[i] <- idiosyncratic_risk_original 
  
  # 风险分解
  for (j in 1:length(symbols)) {
    # 这里计算每只股票的风险贡献
    stock_weights <- rep(0, length(symbols))
    stock_weights[j] <- 1
    
    # 计算每只股票对最低跟踪误差组合的风险贡献
    performance_summary[i, paste0("Risk_Contribution_", symbols[j])] <- 
      (stock_weights[j] * sqrt(t(stock_weights) %*% cov_matrix %*% stock_weights)) / 
      performance_summary$Total_Risk_Min_TE[i]
    
    # 计算每只股票对总风险的贡献
    marginal_contribution <- min_te_weights[j] * (cov_matrix[j, ] %*% min_te_weights)
    performance_summary[i, paste0("Total_Risk_Contribution_", symbols[j])] <- 
      marginal_contribution / performance_summary$Total_Risk_Min_TE[i]
  }
}

# 汇总风险贡献的数据
risk_contribution_summary <- performance_summary[, c("Date", paste0("Risk_Contribution_", symbols))]
risk_contribution_summary_long <- melt(risk_contribution_summary, id.vars = "Date")

# 总风险贡献数据
total_risk_contribution_summary <- performance_summary[, c("Date", paste0("Total_Risk_Contribution_", symbols))]
total_risk_contribution_summary_long <- melt(total_risk_contribution_summary, id.vars = "Date")

# 风险贡献柱状图
p7 <- ggplot(risk_contribution_summary_long, aes(x = Date, y = value, fill = variable)) +
  geom_col(position = "dodge") +
  labs(title = "Risk Contribution by Stock (Min TE)", y = "Risk Contribution") +
  theme_minimal()

# 总风险贡献柱状图
p_total_risk_contribution <- ggplot(total_risk_contribution_summary_long, aes(x = Date, y = value, fill = variable)) +
  geom_col(position = "dodge") +
  labs(title = "Total Risk Contribution by Stock", y = "Total Risk Contribution") +
  theme_minimal()

# 历史VaR图
p_var_historical <- ggplot(performance_summary, aes(x = Date)) +
  geom_line(aes(y = VaR_Historical_Original, color = "Historical VaR Original")) +
  geom_line(aes(y = VaR_Historical_Min_TE, color = "Historical VaR Min TE")) +
  geom_line(aes(y = VaR_Historical_MVO, color = "Historical VaR MVO")) +
  labs(title = "Value at Risk (Historical)", y = "VaR") +
  theme_minimal()

# 正态分布VaR图
p_var_normal <- ggplot(performance_summary, aes(x = Date)) +
  geom_line(aes(y = VaR_Normal_Original, color = "Normal VaR Original")) +
  geom_line(aes(y = VaR_Normal_Min_TE, color = "Normal VaR Min TE")) +
  geom_line(aes(y = VaR_Normal_MVO, color = "Normal VaR MVO")) +
  labs(title = "Value at Risk (Normal Distribution)", y = "VaR") +
  theme_minimal()

# 绘制图表
# 1. 回报率图
p1 <- ggplot(performance_summary, aes(x = Date)) +
  geom_line(aes(y = Original_Return, color = "Original")) +
  geom_line(aes(y = Min_TE_Return, color = "Min TE")) +
  geom_line(aes(y = MVO_Return, color = "MVO")) +
  labs(title = "Annualized Returns", y = "Return") +
  theme_minimal()

# 2. 年化波动率图
p2 <- ggplot(performance_summary, aes(x = Date)) +
  geom_line(aes(y = Original_Volatility, color = "Original")) +
  geom_line(aes(y = Min_TE_Volatility, color = "Min TE")) +
  geom_line(aes(y = MVO_Volatility, color = "MVO")) +
  labs(title = "Annualized Volatility", y = "Volatility") +
  theme_minimal()

# 3. 年化夏普率图
p3 <- ggplot(performance_summary, aes(x = Date)) +
  geom_line(aes(y = Original_Sharpe, color = "Original")) +
  geom_line(aes(y = Min_TE_Sharpe, color = "Min TE")) +
  geom_line(aes(y = MVO_Sharpe, color = "MVO")) +
  labs(title = "Annualized Sharpe Ratio", y = "Sharpe Ratio") +
  theme_minimal()

# 4. 总风险图
p4 <- ggplot(performance_summary, aes(x = Date)) +
  geom_line(aes(y = Total_Risk_Original, color = "Original")) +
  geom_line(aes(y = Total_Risk_Min_TE, color = "Min TE")) +
  geom_line(aes(y = Total_Risk_MVO, color = "MVO")) +
  labs(title = "Total Risk", y = "Risk") +
  theme_minimal()

# 5. 跟踪误差图
p5 <- ggplot(performance_summary, aes(x = Date)) +
  geom_line(aes(y = Tracking_Error_Original, color = "Original")) + 
  geom_line(aes(y = Tracking_Error_Min_TE, color = "Min TE")) +
  geom_line(aes(y = Tracking_Error_MVO, color = "MVO")) +
  labs(title = "Tracking Error", y = "Tracking Error") +
  theme_minimal()

# 6. 系统性风险
p6_systematic <- ggplot(performance_summary, aes(x = Date)) +
  geom_line(aes(y = Systematic_Risk_Min_TE, color = "Min TE")) +
  geom_line(aes(y = Systematic_Risk_MVO, color = "MVO")) +
  geom_line(aes(y = Systematic_Risk_Original, color = "Original")) +  
  labs(title = "Systematic Risk", y = "Risk") +
  theme_minimal()

# 7. 特定风险
p6_idiosyncratic <- ggplot(performance_summary, aes(x = Date)) +
  geom_line(aes(y = Idiosyncratic_Risk_Min_TE, color = "Min TE")) +
  geom_line(aes(y = Idiosyncratic_Risk_MVO, color = "MVO")) +
  geom_line(aes(y = Idiosyncratic_Risk_Original, color = "Original")) +  
  labs(title = "Idiosyncratic Risk", y = "Risk") +
  theme_minimal()

# 输出所有图像
print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
print(p6_systematic)
print(p6_idiosyncratic)
print(p_var_historical)
print(p_var_normal)
print(p7)
# MVO和Min TE优化后的投资组合权重
mvo_weights_history <- data.frame(Date = performance_summary$Date)

# 将MVO权重添加到数据框中
for (i in 1:(length(rebalance_dates) - 1)) {
  current_returns <- returns[(rebalance_dates[i] + 1):rebalance_dates[i + 1], ]
  cov_matrix <- cov(current_returns)
  
  mvo_weights <- mvo_portfolio(cov_matrix)
  mvo_weights_history[i, symbols] <- mvo_weights
}

# 将数据从宽格式转换为长格式
mvo_weights_long <- melt(mvo_weights_history, id.vars = "Date")

# 创建MVO投资组合权重的折线图
p_mvo_weights <- ggplot(mvo_weights_long, aes(x = Date, y = value, color = variable, group = variable)) +
  geom_line(linewidth = 1) +  
  labs(title = "MVO Portfolio Weights", x = "Date", y = "Weights") +
  theme_minimal() +
  scale_color_manual(values = RColorBrewer::brewer.pal(n = length(symbols), name = "Set1")) +  
  theme(legend.title = element_blank())  

# 输出MVO权重的折线图
print(p_mvo_weights)

# Min TE优化后的投资组合权重
min_te_weights_history <- data.frame(Date = performance_summary$Date)

# 将Min TE权重添加到数据框中
for (i in 1:(length(rebalance_dates) - 1)) {
  current_returns <- returns[(rebalance_dates[i] + 1):rebalance_dates[i + 1], ]
  cov_matrix <- cov(current_returns)
  
  min_te_weights <- min_TE_portfolio(original_weights, cov_matrix)
  min_te_weights_history[i, symbols] <- min_te_weights
}

# 将数据从宽格式转换为长格式
min_te_weights_long <- melt(min_te_weights_history, id.vars = "Date")

# 创建Min TE投资组合权重的折线图
p_min_te_weights <- ggplot(min_te_weights_long, aes(x = Date, y = value, color = variable, group = variable)) +
  geom_line(linewidth = 1) +  # 使用linewidth替代size
  labs(title = "Min TE Portfolio Weights", x = "Date", y = "Weights") +
  theme_minimal() +
  scale_color_manual(values = RColorBrewer::brewer.pal(n = length(symbols), name = "Set1")) +  # 自定义颜色
  theme(legend.title = element_blank())  # 清晰的图例

# 输出Min TE权重的折线图
print(p_min_te_weights)

# 计算累计收益
performance_summary$Original_Cumulative_Return <- cumprod(1 + performance_summary$Original_Return / 12) - 1
performance_summary$Min_TE_Cumulative_Return <- cumprod(1 + performance_summary$Min_TE_Return / 12) - 1
performance_summary$MVO_Cumulative_Return <- cumprod(1 + performance_summary$MVO_Return / 12) - 1

# 创建累计收益数据长格式
cumulative_return_history <- performance_summary[, c("Date", "Original_Cumulative_Return", "Min_TE_Cumulative_Return", "MVO_Cumulative_Return")]
cumulative_return_long <- melt(cumulative_return_history, id.vars = "Date")

# 绘制累计收益曲线
p_cumulative_return <- ggplot(cumulative_return_long, aes(x = Date, y = value, color = variable, group = variable)) +
  geom_line(linewidth = 1) +
  labs(title = "Cumulative Returns", x = "Date", y = "Cumulative Return") +
  theme_minimal() +
  scale_color_manual(values = RColorBrewer::brewer.pal(n = 3, name = "Set1")) +
  theme(legend.title = element_blank())

# 输出累计收益曲线
print(p_cumulative_return)

# 计算均值
mean_summary <- data.frame(
  Metric = c(
    "Annualized Return Original",
    "Annualized Return Min TE",
    "Annualized Return MVO",
    "Annualized Volatility Original",
    "Annualized Volatility Min TE",
    "Annualized Volatility MVO",
    "Annualized Sharpe Original",
    "Annualized Sharpe Min TE",
    "Annualized Sharpe MVO",
    "VaR Historical Original",
    "VaR Historical Min TE",
    "VaR Historical MVO",
    "VaR Normal Original",
    "VaR Normal Min TE",
    "VaR Normal MVO",
    "Total Risk Original",
    "Total Risk Min TE",
    "Total Risk MVO",
    "Tracking Error Original",
    "Tracking Error Min TE",
    "Tracking Error MVO",
    "Systematic Risk Original",
    "Systematic Risk Min TE",
    "Systematic Risk MVO",
    "Idiosyncratic Risk Original",
    "Idiosyncratic Risk Min TE",
    "Idiosyncratic Risk MVO"
  ),
  Mean = c(
    mean(performance_summary$Original_Return, na.rm = TRUE),
    mean(performance_summary$Min_TE_Return, na.rm = TRUE),
    mean(performance_summary$MVO_Return, na.rm = TRUE),
    mean(performance_summary$Original_Volatility, na.rm = TRUE),
    mean(performance_summary$Min_TE_Volatility, na.rm = TRUE),
    mean(performance_summary$MVO_Volatility, na.rm = TRUE),
    mean(performance_summary$Original_Sharpe, na.rm = TRUE),
    mean(performance_summary$Min_TE_Sharpe, na.rm = TRUE),
    mean(performance_summary$MVO_Sharpe, na.rm = TRUE),
    mean(performance_summary$VaR_Historical_Original, na.rm = TRUE),
    mean(performance_summary$VaR_Historical_Min_TE, na.rm = TRUE),
    mean(performance_summary$VaR_Historical_MVO, na.rm = TRUE),
    mean(performance_summary$VaR_Normal_Original, na.rm = TRUE),
    mean(performance_summary$VaR_Normal_Min_TE, na.rm = TRUE),
    mean(performance_summary$VaR_Normal_MVO, na.rm = TRUE),
    mean(performance_summary$Total_Risk_Original, na.rm = TRUE),
    mean(performance_summary$Total_Risk_Min_TE, na.rm = TRUE),
    mean(performance_summary$Total_Risk_MVO, na.rm = TRUE),
    mean(performance_summary$Tracking_Error_Original, na.rm = TRUE),
    mean(performance_summary$Tracking_Error_Min_TE, na.rm = TRUE),
    mean(performance_summary$Tracking_Error_MVO, na.rm = TRUE),
    mean(performance_summary$Systematic_Risk_Original, na.rm = TRUE),
    mean(performance_summary$Systematic_Risk_Min_TE, na.rm = TRUE),
    mean(performance_summary$Systematic_Risk_MVO, na.rm = TRUE),
    mean(performance_summary$Idiosyncratic_Risk_Original, na.rm = TRUE),
    mean(performance_summary$Idiosyncratic_Risk_Min_TE, na.rm = TRUE),
    mean(performance_summary$Idiosyncratic_Risk_MVO, na.rm = TRUE)
  )
)

# 输出均值结果
print(mean_summary)