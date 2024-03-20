import streamlit as st
from apps import yolo_client


st.set_page_config(
    page_title="Model Product",
    page_icon="UnionTrain",
    layout="wide")
st.sidebar.title("model Product")

st.sidebar.write("Model product for various algorithm")
st.sidebar.subheader('Navigation:')
selection = st.sidebar.radio("support algorithm:", ["YoLo"])

PAGES = {
    "YoLo": yolo_client,
}

page = PAGES[selection]

page.app()
# getattr(page,"app")