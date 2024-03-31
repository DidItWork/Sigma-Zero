import torch
from network import policyNN
from sim import generate_training_data


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = policyNN(config=dict()).to(device)
    model.train()
    args = {
        'C': 2,
        'num_searches': 3,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 64
    }
    training_data = generate_training_data(model, num_games=3, args=args)
    for epoch in range(3):
        model.backward(training_data)
    torch.save(model.state_dict(), "./test.pt")
    torch.save(model.optimiser.state_dict(), "./opt.pt")
    print("Training complete")


if __name__ == "__main__":
    main()
