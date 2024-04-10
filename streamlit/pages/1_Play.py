import streamlit as st
import numpy as np
import pandas as pd
import time

# TODO Update various fields according to game mode, player type, wins, etc
# TODO If player is human, show the 'Enter your move' input
# TODO If player is AI, show move probabilities and chosen move
# TODO Implement bottom arrow buttons to see past / future moves
# TODO Update chess board image according to move
# TODO Implement 'save game(s)', which saves the game stats to a CSV (e.g. wins/losses, no. of games, no. of turns, etc)
# TODO Implement 'quit', which doesn't save anything, goes back to 'home' page
# TODO Implement pop-up (or other way) to display winner

left_column, center_column, right_column = st.columns([0.2,0.6,0.2], gap='large')

# Left column
with left_column:
    st.subheader("White")
    st.caption("<player type>")
    st.caption("Wins: 0")
    st.divider()
    st.write("Valid moves")
    st.caption("E2E3 \n A2A4 \n D2D3")
    st.divider()
    st.text_input("Enter your move", key="white_move")

# Center column
with center_column:
    st.title("<game mode>")
    st.image('/Users/abramtan/Developer/Sigma-Zero/temp.svg', use_column_width='always')
    c0, c1, c2, c3, c4 = st.columns([0.17,0.1,0.46,0.1,0.17], gap='small')
    c0.button('<<', key='c0_button')
    c1.button('<', key='c1_button')
    c2.caption('Current move: White\nTotal moves: 43')
    c3.button('\>', key='c3_button')
    c4.button('\>>', key='c4_button')

# Right column
with right_column:
    st.subheader("Black")
    st.caption("<player type>")
    st.caption("Wins: 0")
    st.divider()
    st.write("Valid moves")
    st.caption("E2E3 \n A2A4 \n D2D3")
    st.divider()
    st.text_input("Enter your move", key="black_move")

# Sidebar
st.sidebar.button('Refresh')
st.sidebar.button('Save game(s)')
st.sidebar.button('Quit')