import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import google.generativeai as genai

st.set_page_config(page_title="Prueba de Hipotesis", layout="wide")
st.title("App de Prueba de Hipotesis")

st.sidebar.header("Configuracion")
api_key = st.sidebar.text_input("Ingresa tu API Key de Gemini", type="password")

st.sidebar.header("Carga de Datos")
opcion_datos = st.sidebar.radio("Fuente de datos", ("Sintetica", "Archivo CSV"))

if opcion_datos == "Sintetica":
    n_muestras = st.sidebar.number_input("Tamano de la muestra (n)", min_value=30, value=100)
    media_real = st.sidebar.number_input("Media real", value=50.0)
    desviacion_real = st.sidebar.number_input("Desviacion estandar", value=10.0)
    
    if st.sidebar.button("Generar Datos"):
        datos = np.random.normal(loc=media_real, scale=desviacion_real, size=n_muestras)
        df = pd.DataFrame({"Valor": datos})
        st.session_state["df"] = df

elif opcion_datos == "Archivo CSV":
    archivo_subido = st.sidebar.file_uploader("Sube tu CSV", type=["csv"])
    if archivo_subido is not None:
        df = pd.read_csv(archivo_subido)
        st.session_state["df"] = df

if "df" in st.session_state:
    df = st.session_state["df"]
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())
    
    columna_analisis = st.selectbox("Variable a analizar", df.columns)
    datos_analisis = df[columna_analisis]
    
    st.subheader("Visualizacion de Distribuciones")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(datos_analisis, kde=True, ax=ax)
        ax.set_title("Histograma y KDE")
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x=datos_analisis, ax=ax)
        ax.set_title("Boxplot")
        st.pyplot(fig)

    st.subheader("Pruebas Estadisticas (Prueba Z)")
    col3, col4 = st.columns(2)
    
    with col3:
        mu_0 = st.number_input("Hipotesis nula H0 (Media hipotetica)", value=50.0)
        tipo_prueba = st.selectbox("Hipotesis alternativa H1", ["Bilateral", "Cola izquierda", "Cola derecha"])
        
    with col4:
        alpha = st.selectbox("Nivel de significancia (alpha)", [0.01, 0.05, 0.10], index=1)
        varianza_pob = st.number_input("Varianza poblacional conocida", min_value=0.1, value=100.0)

    if st.button("Ejecutar Prueba Z"):
        n = len(datos_analisis)
        x_bar = datos_analisis.mean()
        sigma = np.sqrt(varianza_pob)
        
        z_stat = (x_bar - mu_0) / (sigma / np.sqrt(n))
        
        if tipo_prueba == "Bilateral":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            z_crit_der = stats.norm.ppf(1 - alpha / 2)
            z_crit_izq = -z_crit_der
            rechazar = abs(z_stat) > z_crit_der
        elif tipo_prueba == "Cola derecha":
            p_value = 1 - stats.norm.cdf(z_stat)
            z_crit = stats.norm.ppf(1 - alpha)
            rechazar = z_stat > z_crit
        else:
            p_value = stats.norm.cdf(z_stat)
            z_crit = stats.norm.ppf(alpha)
            rechazar = z_stat < z_crit

        st.write(f"**Estadistico Z calculado:** {z_stat:.4f}")
        st.write(f"**p-value:** {p_value:.4f}")
        
        decision_texto = "Se RECHAZA la hipotesis nula (H0)" if rechazar else "NO se rechaza la hipotesis nula (H0)"
        
        if rechazar:
            st.error(f"Decision: {decision_texto}.")
        else:
            st.success(f"Decision: {decision_texto}.")

        fig_z, ax_z = plt.subplots()
        x_val = np.linspace(-4, 4, 1000)
        y_val = stats.norm.pdf(x_val, 0, 1)
        ax_z.plot(x_val, y_val, color='blue')

        if tipo_prueba == "Bilateral":
            ax_z.fill_between(x_val, y_val, where=(x_val > z_crit_der) | (x_val < z_crit_izq), color='red', alpha=0.5)
        elif tipo_prueba == "Cola derecha":
            ax_z.fill_between(x_val, y_val, where=(x_val > z_crit), color='red', alpha=0.5)
        else:
            ax_z.fill_between(x_val, y_val, where=(x_val < z_crit), color='red', alpha=0.5)

        ax_z.axvline(z_stat, color='black', linestyle='dashed', linewidth=2)
        ax_z.set_title("Curva normal estandar (Z) y region de rechazo")
        st.pyplot(fig_z)

        st.subheader("Modulo de IA (Interpretacion de Gemini)")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                prompt_ia = f"Se realizo una prueba Z con los siguientes parametros: media muestral = {x_bar:.4f}, media hipotetica = {mu_0}, n = {n}, desviacion estandar poblacional = {sigma:.4f}, alpha = {alpha}, tipo de prueba = {tipo_prueba}. El estadistico Z fue = {z_stat:.4f} y el p-value = {p_value:.4f}. Decision matematica: {decision_texto}. ¿Que podemos inferir de estos resultados de forma sencilla? Explica la decision."
                
                st.write("**Prompt enviado:**")
                st.info(prompt_ia)
                
                with st.spinner("Consultando a Gemini..."):
                    respuesta = model.generate_content(prompt_ia)
                    st.write("**Respuesta de la IA:**")
                    st.success(respuesta.text)
            except Exception as e:
                st.error(f"Error al conectar con la API: {e}")
        else:
            st.warning("Ingresa tu API Key en la barra lateral para ver la interpretacion de la IA.")