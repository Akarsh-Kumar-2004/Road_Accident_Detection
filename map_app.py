import streamlit as st
import folium
from streamlit_folium import st_folium

CAMERA_LOCATION = [26.9124, 75.7873]

st.title("🚨 Accident Camera Map")

m = folium.Map(location=CAMERA_LOCATION, zoom_start=15)

folium.Marker(
    CAMERA_LOCATION,
    popup="CAM_543 - Accident Zone",
    tooltip="Camera Active"
).add_to(m)

st_folium(m, width=700, height=500)