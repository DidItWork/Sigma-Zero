import torch
from network import policyNN
from chess_tensor import ChessTensor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = policyNN(config=dict()).to(device)
    network.eval()
    game = ChessTensor()

    board = game.get_representation().unsqueeze(0).cuda()

    policy, value = network(board)

    print(policy)
    print(value)

    network.load_state_dict(torch.load("./test.pt"))
    network.optimiser.load_state_dict(torch.load("./opt.pt"))
    network.eval()
    policy, value = network(board)

    print(policy)
    print(value)
    print("Test complete")


if __name__ == "__main__":
    main()
