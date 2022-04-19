library(palmerpenguins)
data(package = 'palmerpenguins')
head(penguins)
head(penguins_raw)
str(penguins)
library(tidyverse)
penguins %>% 
  count(species)
penguins %>% 
  group_by(species) %>% 
  summarize(across(where(is.numeric), mean, na.rm = TRUE))