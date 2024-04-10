import streamlit as st
import numpy as np
import pandas as pd
import time

# TODO Get the various game options from the user
# TODO When 'start' button is pressed, instantiate a new game, go to 'Play' page

st.header("SigmaZero")
st.divider()
st.selectbox(
    'Mode',
    ('Vanilla', 'Chess960')
)

st.selectbox(
    'Black',
    ('Human', 'SigmaZero', 'StockFish')
)

st.selectbox(
    'White',
    ('Human', 'SigmaZero', 'StockFish')
)

st.selectbox(
    'Number of Games',
    ('1', '3', '5', '10', '20', '50', '100')
)

st.button('Start')