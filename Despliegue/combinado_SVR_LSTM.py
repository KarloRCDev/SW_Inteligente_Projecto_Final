import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.dates as mdates


def mostrar_pagina_svr_lstm():
    # Título de la aplicación
    st.title("Modelos SVR y LSTM para Predicción de Precios de Acciones")

    # Descripción de la aplicación
    st.write("""
    Esta aplicación permite realizar un análisis de predicción de precios de acciones utilizando un modelo combinado de LSTMy SVR.
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

    # Graficar los precios reales
    st.write("### Gráfico de los precios reales")
    fig, ax = plt.subplots()
    ax.plot(data['Close'], label='Precio de cierre', color='blue')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio')
    ax.set_title(f'Precio de Cierre de {ticker} a lo largo del tiempo')
    ax.legend()
    st.pyplot(fig)

    # Preparar los datos
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

    # Optimizar y Entrenar Modelo SVR
    def optimize_svm(X_train, y_train):
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [
            0.1, 0.01, 0.001], 'kernel': ['rbf']}
        grid = GridSearchCV(SVR(), param_grid, refit=True,
                            cv=TimeSeriesSplit(n_splits=5))
        grid.fit(X_train, y_train.ravel())
        return grid.best_estimator_

    svr_model = optimize_svm(X_train, y_train)

    # Entrenar Modelo LSTM
    def train_lstm(X_train, y_train, input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.2, verbose=0)
        return model

    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    lstm_model = train_lstm(X_train_lstm, y_train, (1, X_train.shape[1]))

    # Predicciones
    svr_predictions = scaler_target.inverse_transform(
        svr_model.predict(X_test).reshape(-1, 1))
    lstm_predictions = scaler_target.inverse_transform(
        lstm_model.predict(X_test_lstm))

    combined_predictions = np.median(
        [svr_predictions, lstm_predictions], axis=0)

    # Graficar las predicciones vs los valores reales
    st.write("### Comparación entre Predicciones y Valores Reales (SVR + LSTM)")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index[-len(svr_predictions):], svr_predictions,
            label='Predicciones SVR', color='orange')
    ax.plot(data.index[-len(lstm_predictions):], lstm_predictions,
            label='Predicciones LSTM', color='purple')
    ax.plot(data.index[-len(y_test):], scaler_target.inverse_transform(y_test),
            label='Valores Reales', color='blue')
    ax.set_title('Comparación de Predicciones')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio de Cierre')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Graficar las predicciones vs los valores reales
    st.write("### Comparación entre Predicciones y Valores Reales (Combinado)")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index[-len(combined_predictions):], combined_predictions,
            label='Predicciones Combinadas', color='green')
    ax.plot(data.index[-len(y_test):], scaler_target.inverse_transform(y_test),
            label='Valores Reales', color='blue')
    ax.set_title('Comparación de Predicciones')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Precio de Cierre')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Evaluación del modelo combinado: MAPE y RMSE
    mape_combined = mean_absolute_percentage_error(
        scaler_target.inverse_transform(y_test), combined_predictions)
    rmse_combined = np.sqrt(mean_squared_error(
        scaler_target.inverse_transform(y_test), combined_predictions))

    st.write(f"**MAPE del modelo combinado:** {mape_combined:.4f}")
    st.write(f"**RMSE del modelo combinado:** {rmse_combined:.4f}")
    # Sección para hacer predicciones con datos ingresados
    st.write("### Realizar una Predicción con Nuevos Datos")
    user_input = st.text_input(
        "Ingrese un conjunto de datos (Open, High, Low, Close) separados por coma", "10.0, 11.0, 9.0, 10.5")

    if user_input:
        # Convertir el input a un arreglo de características
        input_data = np.array([float(x)
                              for x in user_input.split(',')]).reshape(1, -1)

        # Normalizar los datos de entrada
        input_scaled = scaler_features.transform(input_data)

        # Preparar datos para LSTM (reshape a 3D)
        input_scaled_lstm = input_scaled.reshape((1, 1, input_scaled.shape[1]))

        # Realizar predicciones
        svr_prediction_scaled = svr_model.predict(input_scaled)
        lstm_prediction_scaled = lstm_model.predict(
            input_scaled_lstm).flatten()

        # Desnormalizar las predicciones
        svr_prediction = scaler_target.inverse_transform(
            svr_prediction_scaled.reshape(-1, 1))
        lstm_prediction = scaler_target.inverse_transform(
            lstm_prediction_scaled.reshape(-1, 1))

        # Combinar las predicciones (promedio o mediana)
        combined_prediction = np.median(
            [svr_prediction[0][0], lstm_prediction[0][0]])

        # Mostrar las predicciones
        st.write(f"**Predicción del modelo SVR:** {svr_prediction[0][0]:.2f}")
        st.write(
            f"**Predicción del modelo LSTM:** {lstm_prediction[0][0]:.2f}")
        st.write(
            f"**Predicción combinada (Mediana):** {combined_prediction:.2f}")


# Llamar a la función de Streamlit para mostrar la página
if __name__ == "__main__":
    mostrar_pagina_svr_lstm()
