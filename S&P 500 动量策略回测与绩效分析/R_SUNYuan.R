# Install necessary packages (if not already installed)
if (!requireNamespace("quantmod", quietly = TRUE)) {
  install.packages("quantmod")
}
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}
if (!requireNamespace("PerformanceAnalytics", quietly = TRUE)) {
  install.packages("PerformanceAnalytics")
}
if (!requireNamespace("lubridate", quietly = TRUE)) {
  install.packages("lubridate")
}
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
if (!requireNamespace("xts", quietly = TRUE)) {
  install.packages("xts")
}
if (!requireNamespace("zoo", quietly = TRUE)) {
  install.packages("zoo")
}

library(quantmod)
library(dplyr)
library(PerformanceAnalytics)
library(lubridate)
library(ggplot2)
library(xts)
library(zoo)

# Get data from Yahoo Finance
getSymbols("^GSPC", from = "2019-01-01", to = "2024-12-31", src = "yahoo")

# Data processing
sp500_data <- data.frame(Date = index(GSPC), coredata(GSPC)) %>%
  mutate(Return = dailyReturn(Ad(GSPC), type = "log")) %>%
  na.omit()

adjusted_close_col <- "GSPC.Adjusted"

# Calculate momentum
sp500_data <- sp500_data %>%
  mutate(Momentum = rollapply(get(adjusted_close_col),
                              width = 252,
                              FUN = function(x) (last(x) - first(x)) / first(x),
                              fill = NA,
                              align = 'right'))

# Generate monthly data
monthly_data <- sp500_data %>%
  filter(!is.na(Momentum)) %>%
  group_by(month = floor_date(Date, "month")) %>%
  summarize(Monthly_Return = prod(1 + Return) - 1,
            Momentum = last(Momentum)) %>%
  ungroup()

# Calculate turnover (ensure non-negative)
monthly_data <- monthly_data %>%
  arrange(month) %>%
  mutate(Turnover = abs(lag(Momentum, order_by = month)))  # Use absolute value of Momentum

# Rolling window size
window_size <- 6

monthly_data <- monthly_data %>%
  arrange(month) %>%
  mutate(
    rolling_IC = rollapplyr(1:nrow(monthly_data),
                            width = window_size,
                            FUN = function(i) {
                              if (i[1] <= window_size) return(NA)
                              cor(monthly_data$Momentum[(i[1]-window_size+1):i[1]],
                                  monthly_data$Monthly_Return[(i[1]-window_size+1):i[1]],
                                  use = "complete.obs")
                            },
                            fill = NA),
    rolling_Hit_Rate = rollapplyr(1:nrow(monthly_data),
                                  width = window_size,
                                  FUN = function(i) {
                                    if (i[1] <= window_size) return(NA)
                                    # 跑赢和跑输基准 (设置基准为0)
                                    hits = sum(monthly_data$Monthly_Return[(i[1]-window_size+1):i[1]] > 0)
                                    losses = sum(monthly_data$Monthly_Return[(i[1]-window_size+1):i[1]] <= 0)
                                    total = hits + losses
                                    # Hit Rate 计算公式
                                    hit_rate = ifelse(total > 0, (hits - losses) / total * 100, NA)
                                    return(hit_rate)  # 返回 Hit Rate
                                  },
                                  fill = NA)
  )

# Calculate annualized return and risk-adjusted return (Sharpe Ratio)
monthly_data <- monthly_data %>%
  mutate(Annualized_Return = (1 + Monthly_Return) ^ 12 - 1,
         Risk_Free_Rate = 0.01 / 12)  # Assume annual risk-free rate is 1%

# Calculate rolling Sharpe ratio
monthly_data <- monthly_data %>%
  mutate(Excess_Return = Monthly_Return - Risk_Free_Rate) %>%
  mutate(Rolling_Sharpe_Ratio = rollapply(Excess_Return,
                                          width = window_size,
                                          FUN = function(x) {
                                            if (sd(x, na.rm = TRUE) == 0) {
                                              return(NA)
                                            }
                                            mean(x, na.rm = TRUE) / sd(x, na.rm = TRUE) * sqrt(12)
                                          },
                                          by.column = TRUE,
                                          fill = NA,
                                          align = 'right'))

# Plotting
# Plot annualized return
p1 <- ggplot(monthly_data, aes(x = month, y = Annualized_Return)) +
  geom_line(color = "blue", size = 1) +
  labs(title = "Annualized Return",
       x = "Month",
       y = "Annualized Return") +
  theme_minimal()

# Plot rolling Sharpe ratio
p2 <- ggplot(monthly_data, aes(x = month, y = Rolling_Sharpe_Ratio)) +
  geom_line(color = "red", size = 1) +
  labs(title = "Rolling Sharpe Ratio",
       x = "Month",
       y = "Sharpe Ratio") +
  theme_minimal()

# Plot turnover as a bar chart
p3 <- ggplot(monthly_data, aes(x = month)) +
  geom_bar(aes(y = Turnover, fill = "Turnover"),
           stat = "identity", position = "dodge", alpha = 0.7) +
  labs(title = "Turnover",
       x = "Month",
       y = "Percentage") +
  scale_fill_manual("", values = c("green")) +
  theme_minimal()

# Plot hit rate as a bar chart
p4 <- ggplot(monthly_data, aes(x = month)) +
  geom_bar(aes(y = rolling_Hit_Rate, fill = "Hit rate"),
           stat = "identity", position = "dodge", alpha = 0.7) +
  labs(title = "Hit rate",
       x = "Month",
       y = "Percentage") +
  scale_fill_manual("", values = c("orange")) +
  theme_minimal()

# Plot information coefficient (IC) as a bar chart
p5 <- ggplot(monthly_data, aes(x = month)) +
  geom_bar(aes(y = rolling_IC * 100, fill = "Information Coefficient (IC)"),
           stat = "identity", position = "dodge", alpha = 0.7) +
  labs(title = "Information Coefficient (IC)",
       x = "Month",
       y = "Percentage") +
  scale_fill_manual("", values = c("purple")) +
  theme_minimal()

# Display the plots
print(p1)  # Annualized return plot
print(p2)  # Rolling Sharpe ratio plot
print(p3)  # Turnover plot
print(p4)  # Hit rate plot
print(p5)  # Information coefficient (IC) plot