library(ggplot2)
library(dplyr)
library(forcats)
library(quantmod)
library(zoo)

setwd("D:/RStudio/dataset")
data1 <- read.csv("portfolio_data.csv")

# Convert the "Date" column to the desired format "2013-05-01"
data1$Date <- as.Date(data1$Date, format = "%m/%d/%Y")

returns1 <- Return.portfolio(data1)


# Convert the xts object to a dataframe and extract the Date column
returns_df <- data.frame(Date = index(returns1), portfolio.returns = coredata(returns1))

# Create a line chart
ggplot(data = returns_df, aes(x = Date, y = portfolio.returns)) +
  geom_line() +
  labs(title = "Portfolio Returns Over Time",
       x = "Date",
       y = "Portfolio Returns")
