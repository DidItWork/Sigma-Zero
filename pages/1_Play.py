import streamlit as st
import pandas as pd
import chess

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

    # old submit function
    # def submit():
    #     st.session_state['current_move'] = st.session_state['human_move']
    #     current_move = st.session_state['current_move']
    #     try:
    #         current_move = chess.Move.from_uci(current_move)
    #         st.session_state['opponent_move'] = st.session_state['game'].play_move(current_move)
    #     except:
    #         st.error(f'{current_move} is invalid.')
    #         print('Invalid move.')
    #     st.session_state['no_of_moves'] += 2
    #     st.session_state['human_move'] = ""

    # new submit function
    def submit():
        user_move_input = st.session_state['human_move']
        current_move = None
        try:
            current_move = chess.Move.from_uci(user_move_input)
        except:
            st.error(f"{user_move_input} is not a valid UCI format. Please check for typos or CAPSLOCK.")
            return
        
        # Retrieve the list of valid moves
        # valid_moves = [str(move) for move in st.session_state['game'].get_move()]
        
        if str(current_move) in valid_moves_list:
            st.session_state['current_move'] = user_move_input
            try:
                st.session_state['opponent_move'] = st.session_state['game'].play_move(current_move)
                st.session_state['no_of_moves'] += 2
            except Exception as e:
                st.error(f"An error occurred when making the move: {e}")
        else:
            st.error(f"{user_move_input} is not a valid move. Please try one of these: {valid_moves_list}")

        st.session_state['human_move'] = ""

    st.text_input("Enter your move (e.g. g1h3)", key="human_move", on_change=submit)
    st.markdown('---')
    st.write("Valid moves")
    valid_moves = st.session_state['game'].get_move()
    valid_moves_list = []
    for move in valid_moves:
        valid_moves_list.append(str(move))
    st.caption(valid_moves_list)
    print(f"No. of moves: {st.session_state['no_of_moves']}")

# Center column
with center_column:
    st.session_state['game'].get_current_board_svg()
    st.title(st.session_state['game_mode'])
    c0, c1, c2, c3, c4 = st.columns([0.5,0.1,0.01,0.2,0.5], gap='small')
    if c1.button('<', key='c1_button'):
        no_of_moves = st.session_state['no_of_moves']
        st.session_state['prev_board_counter'] = min(no_of_moves, st.session_state['prev_board_counter'] + 1)
        st.session_state['game'].get_previous_board_svg(st.session_state['prev_board_counter'])
    if c3.button('\>>', key='c3_button'):
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
if game_status == None:
    st.success(f"Draw!")
    game_results = {
        'sigma_wins': [0],
        'human_wins': [0],
        'draws': [1]
    }
    
elif game_status == chess.WHITE:
    st.success(f'White won!')
    if human_color == 'White':
        game_results = {
            'sigma_wins': [0],
            'human_wins': [1],
            'draws': [0]
        }
    else:
        game_results = {
            'sigma_wins': [1],
            'human_wins': [0],
            'draws': [0]
        }
elif game_status == chess.BLACK:
    st.success(f'Black won!')
    if human_color == 'Black':
        game_results = {
            'sigma_wins': [0],
            'human_wins': [1],
            'draws': [0]
        }
    else:
        game_results = {
            'sigma_wins': [1],
            'human_wins': [0],
            'draws': [0]
        }
else:
    print(f'Game still ongoing.')
    print(f'--------------------')
    game_results = {
            'sigma_wins': [0],
            'human_wins': [0],
            'draws': [0]
        }

# Sidebar
st.sidebar.button('Refresh')
if st.sidebar.button('Save game(s)'):
    df = pd.read_csv('game_stats.csv', index_col=0)
    df = df.add(pd.DataFrame(game_results))
    print(df)
    df.to_csv('game_stats.csv', index=True, header=True)
    st.success(f'Game stats saved.')
if st.sidebar.button('Quit'):
    st.switch_page("Home.py")