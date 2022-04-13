library(palmerpenguins)
data(package = 'palmerpenguins')
head(penguins)
head(penguins_raw)
str(penguins)
library(tidyverse)
penguins %>% 
  count(species)