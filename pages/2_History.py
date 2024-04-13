import streamlit as st
import numpy as np
import pandas as pd
import time

# TODO Implement visualizations of various game stats based on the CSV created by the 'save game(s)' button in 'play' page
# TODO Implement download CSV of game histories

############# DUMMY DATA#############
# COMMENT/UNCOMMENT TO SEE THE DIFFERENCE
# st.session_state.df = pd.DataFrame(np.random.randn(20, 2), columns=["x", "y"])
data = {
    'sigma_wins': [17],
    'human_wins': [3],
    'draws': [5]
}

df = pd.DataFrame(data)

st.session_state.df = df


if "df" not in st.session_state:
    st.header("No data to display here right now.")
    st.markdown('---')
    st.write("Please play a game , and save it first.")

else:
    csv = st.session_state.df.to_csv(index=False).encode('utf-8')

    st.subheader("See how you perfrom against SigmaZero")
    st.markdown('---')
    # st.scatter_chart(st.session_state.df, x="x", y="y")
    
    # Create bar graphs for White and Black
    categories_order = ['SigmaZero Wins', 'Human Wins', 'Draws']

    # # Create bar graphs for White and Black
    # c0, c1 = st.columns([0.5, 0.5])

    # Create a categorical type with the desired order
    ordered_cat = pd.Categorical(
        ['SigmaZero Wins', 'Human Wins', 'Draws'],
        categories=categories_order,
        ordered=True
    )

    # with c0:
    st.header("Game History")
    st.markdown('Here are the stats of the games you have played so far.')
    # Apply the categorical ordering and plot
    white_data = df[['sigma_wins', 'human_wins', 'draws']].T
    white_data.index = ordered_cat
    st.bar_chart(white_data)

    # with c1:
    #     st.subheader("Black (Human)")
    #     # Apply the categorical ordering and plot
    #     black_data = df[['sigma_wins', 'human_wins', 'draws']].T
    #     black_data.index = ordered_cat
    #     st.bar_chart(black_data)

    st.write(st.session_state.df)

    # Sidebar
    st.sidebar.button('Refresh')
    filename = './temp.webp'
    with open(filename, 'rb') as f:
        s = f.read()
    st.sidebar.download_button(label=f'Download CSV', data=csv, file_name='game_stats.csv', mime='text/csv')