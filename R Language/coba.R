library(ggplot2)
library(dplyr)
library(forcats)

setwd("D:/RStudio/dataset")
data <- read.csv("ruu_sql2.csv")

# Create the countplot
ggplot(data, aes(x = fct_infreq(sponsor))) +
  geom_bar(stat = "count")

# Customized countplot
ggplot(data, aes(x = fct_infreq(sponsor), fill = sponsor)) +
  geom_bar() +
  labs(title = "Countplot of Sponsor",
       x = "Sponsor",
       y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


