# Equipo 3-4
# Apodaca Caixba Raquel
# Castillo Inzunza Ernesto
# Quintero Mata Victor Adrian
# Zavala Martinez Bryan Tadeo
# 1. Objetivo del proyecto
# El objetivo es segmentar clientes de una tienda en grupos con características similares utilizando el algoritmo K-means clustering.
# Los datos utilizados fueron:
# Edad
# Ingreso mensual
# Gasto mensual
# Visitas mensuales
#
# Librerias
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Lemos los datos
df = pd.read_csv("clientes_kmeans.csv")

print(df.head())
# Variables
X = df[["Edad", "Ingreso_Mensual", "Gasto_Mensual", "Visitas_Mensuales"]]

# Estandarizacion
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Grafico del codo
inercias = []

for k in range(1, 11):
    modelo = KMeans(n_clusters=k, random_state=42)
    modelo.fit(X_scaled)
    inercias.append(modelo.inertia_)

plt.plot(range(1, 11), inercias, marker="o")
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.title("Método del Codo")
plt.savefig("Codo")

# k-means
kmeans = KMeans(n_clusters=3, random_state=42)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print(df.iloc[35:45])

plt.figure(figsize=(8, 6))

plt.scatter(df["Ingreso_Mensual"], df["Gasto_Mensual"], c=df["Cluster"])

plt.xlabel("Ingreso Mensual")
plt.ylabel("Gasto Mensual")
plt.title("Clusters de Clientes")

plt.savefig("k-means")

# Conclusión
# El algoritmo K-means clustering permitió identificar diferentes tipos de clientes automáticamente.
# Gracias a esto, la tienda puede:
# mejorar sus ventas,
# optimizar campañas de marketing,
# entender mejor el comportamiento de sus clientes,
# y tomar decisiones basadas en datos.
