import torch
from network import policyNN
from chess_tensor import ChessTensor, tensorToAction


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = policyNN(config=dict()).to(device)
    network.eval()
    game = ChessTensor()

    board = game.get_representation().unsqueeze(0).cuda()

    policy, value = network(board)

    print(policy)
    print(value)

    network.load_state_dict(torch.load("./test3.pt"))
    # network.optimiser.load_state_dict(torch.load("./opt3.pt"))
    network.eval()
    policy, value = network(board)

    actions = tensorToAction(policy.squeeze(0))

    print(policy)
    print(actions)
    print(value)
    print("Test complete")


if __name__ == "__main__":
    main()
