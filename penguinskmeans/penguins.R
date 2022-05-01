library(tidyverse)
library(janitor)
library(palmerpenguins)
library(knitr)

dataset <- penguins_raw %>%
    clean_names()

library(skimr)
skim(dataset)

library (GGally)
ggpairs(
  data = dataset,
  columns = c(10:14),
  diag = list(continuous = wrap("barDiag", color = "blue", size =4)),
  upper = list(continuous = wrap("cor", size = 4, bins = 60))
)

penguins <- dataset %>%
  rename (
    bill_length = culmen_length_mm,
    bill_depth = culmen_depth_mm,
    flipper_length = flipper_length_mm,
    body_mass = body_mass_g
  ) %>%
  mutate (
    id = row_number(),
    species = word (species, 1),
    bill_length = scale(bill_length),
    bill_depth = scale(bill_depth),
    flipper_length = scale(flipper_length)
  ) %>%
  select (id, species, island, sex, bill_length, bill_depth, flipper_length, body_mass) %>%
  drop_na (sex)

library(factoextra)
library(FactoMineR)

penguins_PCA <-PCA(penguins[5:7], graph = F)
fviz_screeplot(penguins_PCA)


fviz_pca_biplot(penguins_PCA, geom = "point") +
  geom_point (alpha = 0.2)


## Primer metodo: visual inspection
library(patchwork)
library(glue)
library(here)

kmeans_flex <- function (k) {
  penguins_kmeans <- kmeans(penguins[5:7], k) 
  fviz_cluster(penguins_kmeans, geom = "point", data = penguins[5:7]) +
    labs(title = glue("{k} clusters")) +
    theme (
      plot.background = element_blank(),
      panel.background = element_blank(),plot.title = element_text (margin = margin(0,0,5,0), hjust = 0.5, size = 12, color = "grey", family = "Lato"),
      legend.text = element_text(hjust = 0, size = 8, family = "Lato"),
      legend.position = "none",
      legend.title = element_text(size = 8),
      axis.title = element_text (size = 8),
      axis.text = element_text (size = 8)
    )
}

cluster_possibles <- map (1:9, kmeans_flex)

cluster_possibles[[1]] + cluster_possibles[[2]] + cluster_possibles[[3]] +
  cluster_possibles[[4]] + cluster_possibles[[5]] + cluster_possibles[[6]] +
  cluster_possibles[[7]] + cluster_possibles[[8]] + cluster_possibles[[9]] +
  plot_annotation (
    title = "Kmeans Clustering of Penguins across potential number of clusters \U0022k\U0022 ",
    caption = "Visualization: Joel Soroos @soroosj  |  Data: R palmerpenguins package via R4DS Tidy Tuesday",
    theme = theme (
      plot.title = element_text(hjust = 0.5, vjust = 0.5, size = 14, face = "bold", margin = margin (0,0,20,0)),
      plot.caption = element_text (hjust = 1, size = 7, margin = margin (15,0,0,0)) 
    )
  )

## Metodo dos: Elbow Method

methodologies <- c("wss", "silhouette", "gap_stat")
cluster_optimal <- map (methodologies, ~fviz_nbclust (penguins[5:7], kmeans, method = .x))
cluster_optimal[[1]]

## Metodo tres: Silhouette Method
cluster_optimal[[2]]

## Metodo cuatro: Gap Statistic
cluster_optimal[[3]]

## Metodo cinco: El metodo definitivo, porque, porque prueba con 30 metodologias
library(NbClust)

cluster_30_indexes <- NbClust(data = penguins[,c(5:7)], 
                              distance = "euclidean", 
                              min.nc = 2, max.nc = 9, 
                              method = "complete", 
                              index = "all")


fviz_nbclust (cluster_30_indexes) +
  theme_minimal() +
  labs(title = "Frequency of Optimal Clusters using 30 indexes in NbClust Package")




