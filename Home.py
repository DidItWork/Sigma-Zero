import streamlit as st
from play import PlayTensor
import chess

st.header("SigmaZero")
st.markdown('Try to beat SigmaZero in a game of chess!')
st.markdown('You can try playing vanilla chess, where your usual strategies can be applied, or challenge yourself with Chess960, where the starting positions of the pieces are randomized.')
st.markdown('---')

# Define game options
mode = st.selectbox(
    'Mode',
    ('Vanilla Chess', 'Chess960')
)

color = st.selectbox(
    'Color',
    ('White', 'Black')
)

if color == 'White':
    white_player = 'Human'
    black_player = 'SigmaZero'
else:
    white_player = 'SigmaZero'
    black_player = 'Human'

# Action to perform when 'Start' button is clicked
if st.button('Start'):
    # Store game settings in a session state to preserve them across reruns
    st.session_state['game_mode'] = mode
    st.session_state['color'] = color
    st.session_state['no_of_moves'] = 0
    st.session_state['opponent_move'] = 'None'
    st.session_state['prev_board_counter'] = 0
    st.session_state['current_move'] = ""

    # display a success message
    st.success(f"Game is starting with {white_player} (White) vs {black_player} (Black) in {mode} mode!")

    if mode == 'Chess960':
        mode = True
    else:
        mode = False

    if color == 'White':
        color = chess.WHITE
    else:
        color = chess.BLACK
        
    st.session_state['game'] = PlayTensor()
    st.session_state['game'].start_new_game(chess960=mode, color=color)

    # time.sleep(2)  # Simulate time delay for game setup

    # Redirect to 'Play' page
    st.switch_page("pages/1_Play.py") 

# Check if game settings are stored and display them
if 'game_mode' in st.session_state:
    st.write("Current Game Settings:")
    st.write(f"Mode: {st.session_state['game_mode']}")
    st.write(f"Color: {st.session_state['color']}")