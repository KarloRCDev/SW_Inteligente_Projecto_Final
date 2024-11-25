import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.linalg import pinv
import seaborn as sns
import matplotlib.dates as mdates

# Definir la clase RBF para el modelo


class RBF:
    def __init__(self, num_hidden, sigma=1.0):
        self.num_hidden = num_hidden
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _gaussian(self, X, centers, sigma):
        return np.exp(-np.linalg.norm(X - centers)**2 / (2 * (sigma ** 2)))

    def _calculate_interpolation_matrix(self, X):
        num_samples = X.shape[0]
        distances = np.zeros((num_samples, self.num_hidden))
        for i in range(num_samples):
            for j in range(self.num_hidden):
                distances[i, j] = np.linalg.norm(X[i] - self.centers[j])
        return np.exp(- (distances ** 2) / (2 * self.sigma ** 2))

    def fit(self, X, y):
        self.centers = X[np.random.choice(
            X.shape[0], self.num_hidden, replace=False)]
        Z = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(pinv(Z), y)

    def predict(self, X):
        Z = self._calculate_interpolation_matrix(X)
        return np.dot(Z, self.weights)

# Función para mostrar la página del modelo RBF


def mostrar_pagina_rbf():
    # Título de la aplicación
    st.title("Modelo RBF para Predicción de Precios de Acciones")

    # Descripción de la aplicación
    st.write("""
    Esta aplicación permite realizar un análisis de predicción de precios de acciones utilizando un modelo de red neuronal RBF.
    Puedes seleccionar el ticker de la acción y el rango de fechas, y ver cómo el modelo predice los precios futuros en comparación con los reales.
    """)

    # Entrada del usuario: ticker de la acción
    ticker = st.text_input("Ingrese el ticker de la acción", "FSM")

    # Explicación sobre el ticker
    st.write("""
    **Ticker:** Un ticker es un símbolo único utilizado para identificar una acción en el mercado. Por ejemplo, 'FSM' corresponde a Fortuna Silver Mines Inc.
    """)

    # Selección de rango de fechas
    start_date = st.date_input("Fecha de inicio", pd.to_datetime("2019-01-01"))
    end_date = st.date_input("Fecha de fin", pd.to_datetime("2023-12-31"))

    # Explicación sobre el rango de fechas
    st.write("""
    **Rango de fechas:** Selecciona el rango de fechas para el cual deseas realizar el análisis.
    """)

    # Descargar los datos de Yahoo Finance
    st.write("### Descargando datos...")
    data = yf.download(ticker, start=start_date, end=end_date)

    # Mostrar los datos descargados
    st.write("### Datos descargados:")
    st.write(data.head())

    # Explicación de los datos descargados
    st.write("""
    Los datos descargados muestran información financiera sobre el instrumento seleccionado.
    Incluyen el precio de apertura (Open), el precio más alto del día (High), el precio más bajo del día (Low), el precio de cierre (Close), el volumen de transacciones (Volume), y el precio ajustado de cierre (Adj Close).
    """)

    # Graficar los precios reales
    st.write("### Gráfico de los precios reales")
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Precio de cierre', color='blue')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio')
    ax.set_title(f'Precio de Cierre de {ticker} a lo largo del tiempo')
    ax.legend()
    st.pyplot(fig)

    # Preparar los datos para el modelo RBF
    features = data[['Open', 'High', 'Low', 'Close']].values
    target = features[:, 3].reshape(-1, 1)

    # Normalización de las características
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    features_scaled = scaler_features.fit_transform(features)
    target_scaled = scaler_target.fit_transform(target)

    # Separar los datos en características y objetivo
    X = features_scaled[:-1]
    y = target_scaled[1:]

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    # Crear el modelo RBF y entrenarlo
    num_hidden = 10
    rbf_model = RBF(num_hidden=num_hidden)
    rbf_model.fit(X_train, y_train)

    # Hacer predicciones
    predictions = rbf_model.predict(X_test)

    # Desnormalizar las predicciones y valores reales
    predictions = scaler_target.inverse_transform(predictions.reshape(-1, 1))
    y_test = scaler_target.inverse_transform(y_test)

    # Graficar las predicciones vs los valores reales
    st.write("### Comparación entre Predicciones y Valores Reales")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index[-len(predictions):], predictions,
            label='Predicciones', color='green')
    ax.plot(data.index[-len(y_test):], y_test,
            label='Valores Reales', color='blue')
    ax.set_title('Predicción vs Real del Precio de Cierre')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio de Cierre')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Evaluación del modelo: MAPE y RMSE
    from sklearn.metrics import mean_absolute_percentage_error
    mape_rbf = mean_absolute_percentage_error(y_test, predictions)
    rmse_rbf = np.sqrt(mean_squared_error(y_test, predictions))

    st.write(f"**MAPE del modelo RBF:** {mape_rbf:.4f}")
    st.write(f"**RMSE del modelo RBF:** {rmse_rbf:.4f}")

    # Sección para hacer predicciones con datos ingresados
    st.write("### Realizar una Predicción con Nuevos Datos")
    # Permitir al usuario ingresar un conjunto de datos
    user_input = st.text_input(
        "Ingrese un conjunto de datos (Open, High, Low, Close) separados por coma", "10.0, 11.0, 9.0, 10.5")

    if user_input:
        # Convertir el input a un arreglo de características
        input_data = np.array([float(x)
                              for x in user_input.split(',')]).reshape(1, -1)

        # Normalizar los datos de entrada
        input_scaled = scaler_features.transform(input_data)

        # Realizar la predicción
        prediction_scaled = rbf_model.predict(input_scaled)

        # Desnormalizar la predicción
        prediction = scaler_target.inverse_transform(
            prediction_scaled.reshape(-1, 1))

        st.write(
            f"La predicción para el siguiente precio de cierre del siguiente día es: {prediction[0][0]}")


# Llamar a la función de Streamlit para mostrar la página
if __name__ == "__main__":
    mostrar_pagina_rbf()
