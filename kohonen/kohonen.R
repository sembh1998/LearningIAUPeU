# paquetes
library(kohonen)

# observamos la estructura del dataset
str(iris)

# preparamos la data para el SOM
iris_SOM <- as.matrix(scale(iris[, -5]))

# Creamos el grid que usaremos para el SOM
iris_grid <- somgrid(xdim = 5, ydim = 5, topo = "hexagonal")

# colocamos la semilla 
set.seed(2021)

# usamos la funcion SOM 
iris_SOM_model <- som(X = iris_SOM, 
                      grid = iris_grid)

# Ploteamos los resultados
# Plot tipo 1: counts
plot(iris_SOM_model, type = "counts")

# Plot tipo 2: heatmap 
plot(iris_SOM_model, type = "property", 
     property = getCodes(iris_SOM_model)[, 2],
     main = colnames(iris)[2])

# Plot tipo 3: fan diagram 
plot(iris_SOM_model, type = "codes")