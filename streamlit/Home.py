import streamlit as st
import numpy as np
import pandas as pd
import time

# TODO Get the various game options from the user
# TODO When 'start' button is pressed, instantiate a new game, go to 'Play' page

st.header("SigmaZero")
st.markdown('---')

# Define game options
mode = st.selectbox(
    'Mode',
    ('Vanilla Chess', 'Chess960')
)

black_player = st.selectbox(
    'Black',
    ('Human', 'SigmaZero', 'StockFish')
)

white_player = st.selectbox(
    'White',
    ('Human', 'SigmaZero', 'StockFish')
)

number_of_games = st.selectbox(
    'Number of Games',
    ('1', '3', '5', '10', '20', '50', '100')
)

# Action to perform when 'Start' button is clicked
if st.button('Start'):
    # Store game settings in a session state to preserve them across reruns
    st.session_state['game_mode'] = mode
    st.session_state['black_player'] = black_player
    st.session_state['white_player'] = white_player
    st.session_state['number_of_games'] = int(number_of_games)

    # display a success message
    st.success(f"Game is starting with {white_player} (White) vs {black_player} (Black) in {mode} mode for {number_of_games} game(s).")

    ######AAABBRAAAAMMMMMMMMMMM... HELP REPLACE THIS ONE !!!!!!!!!!!!!!!!!!!!!!!!
    # Placeholder for actual game initialization #ABRAM PLS REPLACE THIS WITH ACTUAL GAME INITIALIZATION
    time.sleep(2)  # Simulate time delay for game setup

    # Redirect to 'Play' page
    st.switch_page("pages/1_Play.py") 

# Check if game settings are stored and display them
if 'game_mode' in st.session_state:
    st.write("Current Game Settings:")
    st.write(f"Mode: {st.session_state['game_mode']}")
    st.write(f"Black: {st.session_state['black_player']}")
    st.write(f"White: {st.session_state['white_player']}")
    st.write(f"Number of Games: {st.session_state['number_of_games']}")
