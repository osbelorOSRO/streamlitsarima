import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from datetime import datetime, timedelta

class TimeSeriesForecaster:
    def __init__(self, data):
        self.data = pd.Series(data)
    
    def fit_sarima(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """
        Ajusta un modelo SARIMA a los datos
        order: (p, d, q) - parámetros no estacionales
        seasonal_order: (P, D, Q, s) - parámetros estacionales
        """
        model = SARIMAX(
            self.data,
            order=order,
            seasonal_order=seasonal_order
        )
        self.fitted_model = model.fit()
        return self.fitted_model
    
    def forecast(self, periods=12):
        """
        Genera pronósticos para los siguientes períodos
        """
        forecast = self.fitted_model.forecast(periods)
        conf_int = self.fitted_model.get_forecast(periods).conf_int()
        return forecast, conf_int

# Configuración de la página
st.set_page_config(
    page_title="Pronóstico de Demanda SARIMA",
    page_icon="📈",
    layout="wide"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Título y descripción
st.title("📈 Pronóstico de Demanda usando SARIMA")
st.markdown("""
Esta aplicación permite realizar pronósticos de demanda utilizando el modelo SARIMA 
(Seasonal AutoRegressive Integrated Moving Average).

### Instrucciones:
1. Ingresa tus datos históricos en el área de texto (un valor por línea)
2. Ajusta los parámetros del modelo en el panel lateral
3. Selecciona el número de períodos a pronosticar
4. Haz clic en "Generar Pronóstico"
""")

# Sidebar para parámetros
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")
    
    # Formato de fecha
    st.subheader("Formato de Datos")
    frecuencia = st.selectbox(
        "Frecuencia de los datos",
        ["Mensual", "Trimestral", "Anual"],
        index=0
    )
    
    fecha_inicio = st.date_input(
        "Fecha de inicio de los datos",
        datetime.now() - timedelta(days=365)
    )
    
    # Parámetros SARIMA
    st.subheader("📊 Parámetros No Estacionales")
    p = st.slider("p (AR)", 0, 3, 1, help="Orden del término autorregresivo")
    d = st.slider("d (Diferenciación)", 0, 2, 1, help="Orden de diferenciación")
    q = st.slider("q (MA)", 0, 3, 1, help="Orden de media móvil")

    st.subheader("🔄 Parámetros Estacionales")
    P = st.slider("P (AR Estacional)", 0, 2, 1, help="Orden del término autorregresivo estacional")
    D = st.slider("D (Diferenciación Estacional)", 0, 1, 1, help="Orden de diferenciación estacional")
    Q = st.slider("Q (MA Estacional)", 0, 2, 1, help="Orden de media móvil estacional")
    s = st.slider("s (Período Estacional)", 4, 12, 12, help="Longitud del ciclo estacional")

# Input de datos
st.header("📝 Datos de Entrada")
data_input = st.text_area(
    "Ingresa tus datos históricos (un número por línea):",
    """100
120
140
160
130
110
105
125
145
165
135
115
110
130
150
170
140
120""",
    height=200
)

# Períodos a pronosticar
col1, col2 = st.columns(2)
with col1:
    periods = st.number_input(
        "Número de períodos a pronosticar",
        min_value=1,
        max_value=24,
        value=6
    )

with col2:
    intervalo_confianza = st.slider(
        "Nivel de confianza (%)",
        min_value=80,
        max_value=99,
        value=95,
        step=1
    )

if st.button("🚀 Generar Pronóstico"):
    try:
        with st.spinner('Procesando datos y generando pronóstico...'):
            # Convertir datos de entrada
            historical_data = [float(x) for x in data_input.strip().split('\n')]
            
            # Crear fechas
            if frecuencia == "Mensual":
                fechas = pd.date_range(fecha_inicio, periods=len(historical_data), freq='M')
            elif frecuencia == "Trimestral":
                fechas = pd.date_range(fecha_inicio, periods=len(historical_data), freq='Q')
            else:
                fechas = pd.date_range(fecha_inicio, periods=len(historical_data), freq='Y')
            
            # Crear y ajustar el modelo
            forecaster = TimeSeriesForecaster(historical_data)
            model = forecaster.fit_sarima(
                order=(p, d, q),
                seasonal_order=(P, D, Q, s)
            )
            
            # Generar pronóstico
            forecast, conf_int = forecaster.forecast(periods)
            
            # Fechas para pronóstico
            fechas_forecast = pd.date_range(
                fechas[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq=fechas.freq
            )
            
            # Mostrar resultados en tabs
            tab1, tab2, tab3 = st.tabs(["📊 Gráfico", "📑 Datos", "📈 Estadísticas"])
            
            with tab1:
                # Crear gráfico con Plotly
                fig = go.Figure()
                
                # Datos históricos
                fig.add_trace(go.Scatter(
                    x=fechas,
                    y=historical_data,
                    name="Datos Históricos",
                    line=dict(color='#1f77b4')
                ))
                
                # Pronóstico
                fig.add_trace(go.Scatter(
                    x=fechas_forecast,
                    y=forecast,
                    name="Pronóstico",
                    line=dict(color='#2ca02c')
                ))
                
                # Intervalo de confianza
                fig.add_trace(go.Scatter(
                    x=fechas_forecast.tolist() + fechas_forecast.tolist()[::-1],
                    y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(44, 160, 44, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'Intervalo de Confianza ({intervalo_confianza}%)'
                ))
                
                fig.update_layout(
                    title="Pronóstico de Demanda",
                    xaxis_title="Fecha",
                    yaxis_title="Valor",
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Datos Históricos")
                    hist_df = pd.DataFrame({
                        'Fecha': fechas,
                        'Valor': historical_data
                    })
                    st.dataframe(hist_df, hide_index=True)
                
                with col2:
                    st.subheader("Pronóstico")
                    forecast_df = pd.DataFrame({
                        'Fecha': fechas_forecast,
                        'Pronóstico': forecast.values,
                        'Límite Inferior': conf_int.iloc[:, 0],
                        'Límite Superior': conf_int.iloc[:, 1]
                    })
                    st.dataframe(forecast_df, hide_index=True)
            
            with tab3:
                st.subheader("Estadísticas del Modelo")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("AIC", round(model.aic, 2))
                    st.metric("BIC", round(model.bic, 2))
                
                with col2:
                    st.metric("Log Likelihood", round(model.llf, 2))
                    st.metric("Número de Observaciones", len(historical_data))
                
                st.subheader("Resumen del Modelo")
                st.code(str(model.summary()))
        
        st.success('¡Pronóstico generado exitosamente! 🎉')
        
    except Exception as e:
        st.error(f"Error al procesar los datos: {str(e)}")
        st.info("Por favor, verifica que los datos ingresados sean números válidos y que haya suficientes observaciones para el modelo.")