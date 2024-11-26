from combinado_SVR_LSTM import mostrar_pagina_svr_lstm
from svr import mostrar_pagina_svr
from lstm import mostrar_pagina_lstm
from rfr import mostrar_pagina_rfr
from rbf import mostrar_pagina_rbf
import streamlit as st

# La primera llamada a Streamlit debe ser st.set_page_config()
st.set_page_config(
    page_title="Despliegue web",
    page_icon="👨🏻‍💻",
)

# Importa las funciones de las páginas

# Función para mostrar la página de inicio


def mostrar_pagina_inicio():
    st.sidebar.title('Menú')

    # Widget interactivo para seleccionar la página
    page = st.sidebar.radio(
        'Selecciona una opción del menú',
        [
            "PRESENTACIÓN",
            "MODELO ENSAMBLADO",
            "MODELO LSTM",
            "MODELO SVR",
            "MODELO RBF",
            "MODELO RFR",
        ]
    )

    # Contenido según la página seleccionada
    if page == "PRESENTACIÓN":
        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            st.subheader("UNIVERSIDAD NACIONAL MAYOR DE SAN MARCOS")
            st.subheader("Facultad de Ingeniería de Sistemas e Informática")
            st.write("\n")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.image(
                    "https://seeklogo.com/images/U/universidad-nacional-mayor-de-san-marcos-logo-302291E186-seeklogo.com.png",
                    width=100,
                )
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

    elif page == "MODELO ENSAMBLADO":
        mostrar_pagina_svr_lstm()  # Asegúrate de que esta función está correctamente importada

    elif page == "MODELO LSTM":
        mostrar_pagina_lstm()

    elif page == "MODELO SVR":
        mostrar_pagina_svr()

    elif page == "MODELO RBF":
        mostrar_pagina_rbf()  # Coloca un marcador para la implementación futura

    elif page == "MODELO RFR":
        mostrar_pagina_rfr()


# Punto de entrada
if __name__ == "__main__":
    mostrar_pagina_inicio()
