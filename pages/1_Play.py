import streamlit as st
import numpy as np
import pandas as pd
import time
import chess

# TODO Implement 'save game(s)', which saves the game stats to a CSV (e.g. wins/losses, no. of games, no. of turns, etc)
# TODO Refresh valid moves upon playing a move

left_column, center_column, right_column = st.columns([0.2,0.6,0.2], gap='large')

if st.session_state['color'] == 'White':
    human_color = 'White'
    sigmazero_color = 'Black'
else:
    human_color = 'Black'
    sigmazero_color = 'White'

# Left column
with left_column:
    st.subheader(human_color)
    st.caption('Human')
    st.markdown('---')
    st.write("Valid moves")
    valid_moves = st.session_state['game'].get_move()
    valid_moves_list = []
    for move in valid_moves:
        valid_moves_list.append(str(move))
    st.caption(valid_moves_list)
    st.markdown('---')

    def submit():
        st.session_state['current_move'] = st.session_state['human_move']
        current_move = st.session_state['current_move']
        try:
            current_move = chess.Move.from_uci(current_move)
            st.session_state['opponent_move'] = st.session_state['game'].play_move(current_move)
        except:
            st.error(f'{current_move} is invalid.')
            print('Invalid move.')
        st.session_state['no_of_moves'] += 2
        st.session_state['human_move'] = ""

    st.text_input("Enter your move (e.g. g1h3)", key="human_move", on_change=submit)

    current_move = st.session_state['current_move']
    print(current_move)
    # if current_move:
        # try:
        #     current_move = chess.Move.from_uci(current_move)
        #     st.session_state['opponent_move'] = st.session_state['game'].play_move(current_move)
        # except:
        #     st.error(f'{current_move} is invalid.')
        #     print('Invalid move.')
        # st.session_state['no_of_moves'] += 2
    print(f"No. of moves: {st.session_state['no_of_moves']}")

# Center column
with center_column:
    st.session_state['game'].get_current_board_svg()
    st.title(st.session_state['game_mode'])
    c0, c1, c2, c3, c4 = st.columns([0.5,0.1,0.01,0.1,0.5], gap='small')
    if c1.button('<', key='c1_button'):
        no_of_moves = st.session_state['no_of_moves']
        st.session_state['prev_board_counter'] = min(no_of_moves, st.session_state['prev_board_counter'] + 1)
        st.session_state['game'].get_previous_board_Svg(st.session_state['prev_board_counter'])
    if c3.button('\>', key='c3_button'):
        st.session_state['prev_board_counter'] = 0
    st.image('./board.svg', use_column_width='always')
    print(f"prev_board_counter: {st.session_state['prev_board_counter']}")

# Right column
with right_column:
    st.subheader(sigmazero_color)
    st.caption('SigmaZero')
    st.markdown('---')
    st.write("Previous move")
    st.caption(st.session_state['opponent_move'])
    st.markdown('---')

# Check game status
game_status = st.session_state['game'].check_if_end()
# print(f'{game_status}')
if game_status == None:
    st.success(f"Draw!")
elif game_status == chess.WHITE:
    st.success(f'White won!')
elif game_status == chess.BLACK:
    st.success(f'Black won!')
else:
    print(f'Game still ongoing.')
    print(f'--------------------')

# Sidebar
st.sidebar.button('Refresh')
if st.sidebar.button('Save game(s)'):
    pass
if st.sidebar.button('Quit'):
    st.switch_page("Home.py") 