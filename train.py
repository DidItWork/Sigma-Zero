import torch
from network import policyNN
from sim import generate_training_data


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = policyNN(config=dict()).to(device)
    model.train()
    args = {
        'C': 2,
        'num_searches': 5,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 64
    }
    for epoch in range(5):
        training_data = generate_training_data(model, num_games=5, args=args)
        model.backward(training_data)
        print(f"Epoch {epoch} training complete")
    torch.save(model.state_dict(), "./test3.pt")
    torch.save(model.optimiser.state_dict(), "./opt3.pt")
    print("Training complete")


if __name__ == "__main__":
    main()
