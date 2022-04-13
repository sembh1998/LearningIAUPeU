
library(tidyverse)
library(factoextra)
library(cowplot)
library(ggpubr)
library(cluster)
library(fviz_cluster)

# Escalar y centrar las variables: media=0 y sd= 1 

#El set de datos USArrests contiene información sobre 
#el número de delitos (asaltos, asesinatos y secuestros)
View(USArrests)
?USArrests
summary(USArrests)

inseguridad = scale(USArrests, center = TRUE, scale = TRUE)
summary(inseguridad)

inseguridad = as.data.frame(inseguridad)
ciudades=rownames(inseguridad)


#creamos 4 cluster en funcion a su grado de inseguridad

kmcluster = kmeans(inseguridad,centers=4,nstart = 50)
kmcluster

#graficamos los cluster en funcion del %muertes y %asaltos

inseguridad = inseguridad %>% mutate(cluster = kmcluster$cluster)

(g1=ggplot(inseguridad, aes(x = Murder, y = Assault)) +
  geom_point(aes(color=as.factor(cluster)), size=10)+
  geom_text(aes(label = cluster), size = 5) +
  theme_bw() +
  theme(legend.position = "none")+
  labs(title = "Kmenas con k=4") 
)

#graficamos sus 2 primeras componentes

fviz_cluster(kmcluster, inseguridad)+ theme_minimal()


#fviz_cluster(km.res, iris[, -5], ellipse.type = "norma")


#Adicionamos la etiqueta de las ciudades
rownames(inseguridad)=ciudades

fviz_cluster(kmcluster, inseguridad, show.clust.cent = T,
             ellipse.type = "euclid", star.plot = T, repel = T) +
  labs(title = "Resultados clustering K-means") +
  theme_bw()
 

#### Creamos 2 cluster k=2
kmcluster2 = kmeans(inseguridad, centers=2, nstart = 50)
inseguridad = inseguridad %>% mutate(cluster2 = kmcluster2$cluster)

(g2=ggplot(inseguridad, aes(x = Murder, y = Assault)) +
  geom_point(aes(color=as.factor(cluster2)), size=10)+
  geom_text(aes(label = cluster2), size = 5) +
  theme_bw() +
  theme(legend.position = "none")+
  labs(title = "Kmenas con k=2") 
)

plot_grid(g1,g2)


#::::::::::::::: Numero optimo de Cluster  :::::::::::::::

# creamos una funcion que nos retorne la var.within para cada k
total_within = function(n_clusters, data, iter.max=1000, nstart=50){
  
  cluster_means = kmeans(data,centers = n_clusters,
                       iter.max = iter.max,
                       nstart = nstart)
  return(cluster_means$tot.withinss)
}

# Se aplica esta función con para diferentes valores de k
total_withinss <- map_dbl(.x = 1:15,
                          .f = total_within,
                          data = inseguridad)
total_withinss

#graficamos la varianza total

data.frame(n_clusters = 1:15, suma_cuadrados_internos = total_withinss) %>%
  ggplot(aes(x = n_clusters, y = suma_cuadrados_internos)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = 1:15) +
  labs(title = "Suma total de cuadrados intra-cluster") +
  theme_bw()

#otro metodo, usando el paquete "factoextra"
matriz_dist=get_dist(inseguridad, method = "euclidean")

fviz_nbclust(inseguridad, FUNcluster = kmeans, 
             method = "wss", k.max = 15, 
             diss = matriz_dist, nstart = 50)

