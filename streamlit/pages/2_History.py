import streamlit as st
import numpy as np
import pandas as pd
import time

# TODO Implement visualizations of various game stats based on the CSV created by the 'save game(s)' button in 'play' page
# TODO Implement download CSV of game histories

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])

st.header("SigmaZero vs StockFish")
st.divider()
st.scatter_chart(st.session_state.df, x="x", y="y")

# Sidebar
st.sidebar.button('Refresh')
filename = '/Users/abramtan/Developer/Sigma-Zero/temp.svg'
with open(filename, 'rb') as f:
    s = f.read()
st.sidebar.download_button(label=f'Download CSV', data=s, file_name='temp.svg')