import streamlit as st

from apps import model
from pages import MultiPage

app = MultiPage()

st.set_page_config(layout="wide")
st.title("Portfolio Manager")

app.add_page("Model", model.app)

app.run()
