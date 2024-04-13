import streamlit as st
import numpy as np
import pandas as pd
import time

############# DUMMY DATA#############
# COMMENT/UNCOMMENT TO SEE THE DIFFERENCE
# st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])
# data = {
#     'sigma_wins': [17],
#     'human_wins': [3],
#     'draws': [5]
# }

# df = pd.DataFrame(data)


# csv = st.session_state.df.to_csv(index=False).encode('utf-8')

st.subheader("See how you perfrom against SigmaZero")
st.markdown('please play a game and save the game to see the stats here.')
st.markdown('---')

# Define the desired order (by default, the order is alphabetical)
categories_order = ['SigmaZero Wins', 'Human Wins', 'Draws']

ordered_cat = pd.Categorical(
    ['SigmaZero Wins', 'Human Wins', 'Draws'],
    categories=categories_order,
    ordered=True
)
st.header("Game History")
st.markdown('Here are the stats of the games you have played so far.')
# Apply the categorical ordering and plot
df = pd.read_csv('game_stats.csv', index_col=0)
data = df[['sigma_wins', 'human_wins', 'draws']].T
data.index = ordered_cat
st.bar_chart(data)

st.write(df)

# Sidebar
st.sidebar.button('Refresh')
filename = './game_stats.csv'
with open(filename, 'rb') as f:
    s = f.read()
st.sidebar.download_button(label=f'Download CSV', data=s, file_name='game_stats.csv', mime='text/csv')