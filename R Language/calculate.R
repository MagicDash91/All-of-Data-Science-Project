library(ggplot2)
library(dplyr)
library(forcats)
library(quantmod)
library(zoo)
library(plotly)

setwd("D:/RStudio/dataset")
data3 <- read.csv("portfolio_data.csv")

# Convert the "Date" column to the desired format "2013-05-01"
data3$Date <- as.Date(data3$Date, format = "%m/%d/%Y")

# Convert the data to an xts object using only the numeric columns
prices_xts <- xts(data3[, -1], order.by = data3$Date)

# Calculate the returns for each asset
returns_xts <- Return.calculate(prices_xts)

# Convert the "Date" column to the desired format "2013-05-01"
data3$Date <- as.Date(data3$Date, format = "%m/%d/%Y")

# Convert the data to an xts object using only the numeric columns
prices_xts <- xts(data3[, -1], order.by = data3$Date)

# Calculate the returns for each asset
returns_xts <- Return.calculate(prices_xts)

# Convert the returns back to a data frame
returns_df <- data.frame(Date = index(returns_xts), coredata(returns_xts))

# Create an interactive line chart for each asset's returns
plot_ly(data = returns_df, x = ~Date) %>%
  add_lines(y = ~AMZN, name = "AMZN", line = list(color = "blue")) %>%
  add_lines(y = ~DPZ, name = "DPZ", line = list(color = "green")) %>%
  add_lines(y = ~BTC, name = "BTC", line = list(color = "orange")) %>%
  add_lines(y = ~NFLX, name = "NFLX", line = list(color = "red")) %>%
  layout(title = "Asset Returns Over Time",
         xaxis = list(title = "Date"),
         yaxis = list(title = "Returns"),
         showlegend = TRUE)
