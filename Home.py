import streamlit as st
import matplotlib.pyplot as plt
import plots
from plots import hour_of_day
from data import load_data

st.set_page_config(
    page_title='🚘 Analyse', 
    layout='centered',
    )

df = load_data()

st.header('📊 US-Unfälle Datenanalyse')

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Übersicht",
    "Wetter",
    "Tageszeit",
    "Regionen",
    "Verkehrsobjekte",
    "Heatmap"
])

with tab1:
    st.write(df.head())
    total_entries = df.shape[0]
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Gesamtanzahl Unfälle", f"{total_entries:,}")
    with col2:
        avg_severity = df['Severity'].mean()
        st.metric("Durchschnittliche Schwere", f"{avg_severity:.2f}")
    with col3:
        avg_temp = df['Temperature(F)'].mean()
        st.metric("Ø Temperatur (F)", f"{avg_temp:.1f}°")

with tab2:
    plots.weather(df)

with tab3:
    hour_of_day(df)

with tab4:
    plots.state_analysis(df)

with tab5:
    plots.traffic_features(df)

with tab6:
    plots.heatmap(df)