from rfr import mostrar_pagina_random_forest
from rbf import mostrar_pagina_rbf
from svr import mostrar_pagina_svr
from lstm import mostrar_pagina_lstm
from combinado_SVR_LSTM import mostrar_pagina_svr_lstm
import streamlit as st

# Configuración inicial de la página
st.set_page_config(
    page_title="Proyecto Final - Software Inteligente",
    layout="wide",
)


# Función para mostrar la página de inicio


def mostrar_pagina_presentacion():
    # Crear columnas para centrar el contenido
    # Ajustar proporciones para centrar
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.subheader("UNIVERSIDAD NACIONAL MAYOR DE SAN MARCOS")
        st.subheader("Facultad de Ingeniería de Sistemas e Informática")
        st.write("\n")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col3:
        st.image(
            "https://seeklogo.com/images/U/universidad-nacional-mayor-de-san-marcos-logo-302291E186-seeklogo.com.png", width=200)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.header("Proyecto Final Integral del Curso")
        st.subheader("Curso: Software Inteligente - 2024-2")
        st.subheader("Profesor:")
        st.markdown(
            """ 
            - **Herrera Quispe** - Jose Alfredo 
            """
        )
        st.subheader("Integrantes del equipo:")
        st.markdown(
            """ 
            - **Romero Cisneros** - Karlo Brandi
            - **Hernandez Bianchi** - Stefano Alessandro 
            """
        )


# Diccionario para mapear páginas a funciones
pages = {
    "INICIO": mostrar_pagina_presentacion,
    "MODELO RBF": mostrar_pagina_rbf,
    "MODELO RFR": mostrar_pagina_random_forest,
    "MODELO LSTM": mostrar_pagina_lstm,
    "MODELO SVR": mostrar_pagina_svr,
    "MODELO ENSAMBLADO": mostrar_pagina_svr_lstm,
}

# Controlador principal


def main():
    st.sidebar.title("Menú de Navegación")
    # Configurar "PRESENTACIÓN" como la opción predeterminada (índice 0)
    selected_page = st.sidebar.radio(
        "Selecciona una opción del menú",
        list(pages.keys()),
        index=0  # Índice de la opción "PRESENTACIÓN"
    )

    # Mostrar la página seleccionada
    page_function = pages.get(selected_page)
    if page_function:
        try:
            page_function()
        except Exception as e:
            st.error(f"Ocurrió un error al cargar la página: {e}")
    else:
        st.error("Página no encontrada.")


# Entrada principal
if __name__ == "__main__":
    main()
