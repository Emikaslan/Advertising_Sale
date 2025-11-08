import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Dataset laden
dataset = pd.read_csv("Advertising.csv")

# Unnötige Spalte entfernen
dataset = dataset.drop(columns=["Unnamed: 0"])

# Daten vorbereiten
y = dataset["Sales"]
X = dataset[["TV", "Radio", "Newspaper"]]

# Übersicht der Daten
print("\n=== HEAD ===")
print(dataset.head())

print("\n=== INFO ===")
print(dataset.info())

print("\n=== DESCRIBE ===")
print(dataset.describe())

# Daten aufteilen (Trainings- und Testdaten)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modell erstellen und trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersage
y_pred = model.predict(X_test)

# Bewertung
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R Squared Score: {r2}")

# Diagramm erstellen und speichern
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Ideal fit')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales using Machine Learning")
plt.legend()

# Diagramm speichern
plt.savefig("plot.png")
print("\n✅ Diagramm gespeichert als 'plot.png' – schau rechts in der Dateiliste.")

# Hinweis für Replit (damit plt.show() nicht blockiert)
plt.close()
