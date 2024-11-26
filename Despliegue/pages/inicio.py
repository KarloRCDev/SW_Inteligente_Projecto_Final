import streamlit as st

# Configuración inicial de la página
st.set_page_config(
    page_title="Proyecto Final - Software Inteligente",
    layout="wide",
    page_icon="🎓",
)

# Función para mostrar la página de presentación


def mostrar_pagina_presentacion():
    # Crear columnas para centrar el contenido
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.subheader("UNIVERSIDAD NACIONAL MAYOR DE SAN MARCOS")
        st.subheader("Facultad de Ingeniería de Sistemas e Informática")
        st.write("\n")
        # Centrar el logotipo
        logo_col1, logo_col2, logo_col3 = st.columns([1, 1, 1])
        with logo_col2:
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
            - **Hernández Bianchi** - Stefano Alessandro  
            """
        )

# Controlador principal


def main():
    mostrar_pagina_presentacion()


# Entrada principal
if __name__ == "__main__":
    main()
