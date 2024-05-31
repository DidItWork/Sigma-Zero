# SigmaZero
A PyTorch implementation of Supervised and Reinforcement Learning in game-playing according to Alpha Zero

## Environment Setup

We will be using Miniconda as the environment manager, but you can adapt the steps for any similar tool you might prefer.

Ensure Miniconda is installed on your system. If not installed, you can download it from Miniconda's official website. This project is developed using Python 3.11, so it is advisable to use a compatible version of Miniconda.

Navigate to the root directory of the project file where you can find `environment.yml`. This file lists all the necessary packages and their specific versions required to run the application.

Create the Conda environment using the following command:

```
conda env create -f environment.yml
```

Activate environment

```
conda activate sigmazero
```

If pytorch does not work, try reinstalling it via the following from [pytorch](https://pytorch.org/)

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Setting up Graphical User Interface (GUI)

### Environment Setup and Running Application

Run the Apllication: 

Start the application using Streamlit by running:

`streamlit run Home.py`

### Troubleshooting

1. Dependency Errors: 

    If you encounter errors related to missing packages or version conflicts, ensure that the environment.yml file includes all necessary dependencies with correct versions. Ensure that you are using the right version of streamlit:

    `pip install streamlit==1.33.0`

2. Environment Activation: 

    Make sure you activate the correct Conda environment before attempting to run the application. If the environment name is incorrect, check the name specified in the environment.yml file.

## Training

### Supervised Learning on Vanilla Chess

Download >2000 ELO Player data for Standard Chess (Blitz and Lightning Chess data will contain non-optimal moves) from [FICS](https://www.ficsgames.org/download.html).

Place the downloaded `.pgn.bz2` in the a `saves` folder and set the file path as well as the number of games to generate in `generate_training_supervised.py`

```
python generate_training_supervised.py
```

Train supervised learning model

```
python train_supervised.py
```

### Unsupervised Learning

Reinforcement Learning can be run with the following. 

```
python train_RL.py
```

Hyperparameters for training can be set with the arguments dictionary in the file

```
args = {
        'C': 2,
        'num_searches': 100,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 30,
        'batch_size': 128,
        "start_epoch": 0,
        "chess960": True,
    }
```

## Testing

We test our models against each level of Stockfish in a best of 5 format, results can be found in `logs/log.txt`. You can load your model weights and the path to your Stockfish engine in `eval.py`

You can download the model weights [here](supervised_model_best.pt).

```
python eval.py
```

## Results

Our network was trained on 15000 >2000 ELO Standard Chess data for 60 epochs and the best results are shown in the table below.

The results are obtained from best of 5 games against different levels of Stockfish, a win is awarded 1 point and a draw is awarded 0.5 point. The Sigmazero AI and Stockfish will take turns to play White and Sigmazero will advance to the next level if it gets 2.5 points or more

| StockFish Skill Level | Time Limit | Search Depth | Estimated ELO |
|-----------------------|------------|--------------|---------------|
| 0                     | 1          | 5            | 1376          |
| 1                     | 1          | 5            | 1462          |
| 2                     | 1          | 5            | 1547          |
| 3                     | 1          | 5            | 1596          |
| 4                     | 1          | 5            | 1718          |
| 5                     | 1          | 5            | 1804          |
| 6                     | 1          | 5            | 1993          |
| 7                     | 1          | 5            | 2012          |
| 8                     | 1          | 6            | 2127          |
| 9                     | 2          | 7            | 2270          |
| 20                    | 10         | 50           | 3100          |

### Vanilla Chess

| Model          | MCTS Simulations | SF Level | Estimated ELO | Model Win | Model Loss | Model Draw | Games | Score |
|----------------|------------------|----------|---------------|-----------|------------|------------|-------|-------|
| 48k_supervised | 800              | 3        | 1596          | 3         | 1          | 0          | WLWW  | 3.0   |
| 48k_supervised | 800              | 4        | 1718          | 3         | 0          | 0          | WWW   | 3.0   |
| 48k_supervised | 800              | 5        | 1804          | 2         | 0          | 1          | DWW   | 2.5   |
| 48k_supervised | 800              | 6        | 1993          | 2         | 0          | 1          | DWW   | 2.5   |
| 48k_supervised | 800              | 7        | 2012          | 3         | 1          | 0          | WLWW  | 3.0   |
| 48k_supervised | 800              | 8        | 2127          | 2         | 1          | 1          | WLDW  | 2.5   |
| 48k_supervised | 800              | 9        | 2270          | 0         | 3          | 2          | LDLDL | 1.0   |

### Chess 960

*Coming Soon*

<!-- ![chess_960](/images/chess_960.png) -->

## Logic

### Game Tree Node Attributes
- $N_i$, number of times node has been selected / number of times the node has been through the simulation (integer)
- $W_i$, the sum of expected value of the node (not an integer, "the number of wins for the node")
- $p$, policy values of child nodes
- $s$, representation of board state (8x8xN tensor)

### Alpha Zero MCTS
1. Selection: Start from root node (current game state) and select successive nodes based on Upper Confidence Bound Criterion (UCB) until a leaf node L is reached (a leaf node is any node that has a potential child from which no simulation has yet been initiated) or a terminal node.
$$\text{UCB} = \frac{w_i}{n_i}+p_ic\frac{\sqrt{N_i}}{1+n_i}$$
, where $c$ is a constant, $p_i$ is the policy of the child node and $n_i$ is its simulation count
3. Expansion: Unless L ends the game decisively for either player, randomly initialize an unexplored child node.
4. Backpropagation: Using the value generated by the neural network $f_\theta$, update the N and W values of the current node and all its parent nodes.
5. Repeat steps 1 to 3 for N iterations

### Self-Play and Training
1. Self-Play until the game ends using MCTS and $f_\theta$
2. Store the chosen action taken at each state and the values of the node (-1,0,1) depending on the player and whether he won or lost the game. One training sample should contain: (board state s, the action chosen $\pi$, the value of the node z)
3. Minimize loss function of the training samples in the batch.

$$l = (z-v)^2-\pi^T\log{p}+c||\theta||^2$$

$c$ is a constant

### Board State Representation

For the player's perspectives, this is what the tensor will look like. The board will change according to the current player.

White's View:

![white_view](https://github.com/DidItWork/Sigma-Zero/assets/63920704/39db00c8-c4b2-4578-b308-c185e408f54c)

Black's View:

![black_view](https://github.com/DidItWork/Sigma-Zero/assets/63920704/36f20d8a-d3e8-4731-8c5d-f24864e4eef9)

The board is represented as a (119, 8, 8) tensor, as calculated with MT + L. Where M = 14, T = 8, L = 7.

M represents the number of pieces/planes that are recorded in the board state. In our implementation, we mimicked AlphaZero's implementation of keeping track of all 12 pieces with 2 repetition planes. The order of the planes are as follows:
1. White Pawns
2. White Knights
3. White Bishops
4. White Rooks
5. White Queens
6. White King
7. Planes 7 to 12 are the same as 1 to 6, but for black pieces.
13. 1-fold repetition plane, a constant binary value
14. 2-fold repetition plane

T represents the number of half-steps that are kept track of. In this case we keep track of 8 half-steps, or 4 full turns. The latest update of the half-step is recorded in the first planes.

L is not time-tracked. It is a constant 7 planes that represents special cases of the board regardless of time. The order is as follows:
1. Current player's color
2. Total Moves that have been played to understand depth
3. White King's castling rights
4. White Queen's castling rights
5. Black King's castling rights
6. Black Queen's castling rights
7. No progress plane, for 50-move rule


### Action Representation

The actions are represented with an 8x8x73 tensor which can be flattened into a 4672 vector. The planes of the tensor represent the location on the board from which the chess piece should be picked up from.

- The first 8x7 channels/planes represent the number of squares to move (1 to 7) for the queen/rook/pawn/bishop/king as well as the direction. (Movement of pawn from 7th rank is assumed to be a promotion to queen)
- The next 8 channels/planes represent the direction to move the knight
- The last 9 channels represent the underpromotion of the pawn to knight, bishop, and rook resp. (through moving one step from the 7th rank or a diagonal capture from the 7th rank).
