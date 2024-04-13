import torch
from network import policyNN
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from chess_tensor import actionsToTensor
from train_RL import train, chessDataset

device = "cuda" if torch.cuda.is_available else "cpu"

def main(train_config=None, optimiser=None, lr_scheduler=None):

    batch_size = train_config.get("batch_size", 128)
    num_epoch = train_config.get("epoch", 100)

    dataset = torch.load("train_set.pt")

    train_dataset = chessDataset(training_data=dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size = batch_size,
                                  shuffle=True,
                                  num_workers=4,
                                  collate_fn=train_dataset.collatefn)

    train(model=model,
          dataloader=train_dataloader,
          optimiser=optimiser,
          total_steps=num_epoch,
          lr_scheduler=lr_scheduler)

if __name__ == "__main__":

    train_config = {
        "batch_size": 128,
        "epoch": 1
    }
    model_config = {}

    lr = 0.0001
    lr_step_size = 5000

    model = policyNN(model_config).to(device)

    optimiser = Adam(params=model.parameters(), lr=lr)

    lr_scheduler = StepLR(optimizer=optimiser, step_size=lr_step_size, gamma=0.95)

    main(train_config=train_config, optimiser=optimiser, lr_scheduler=lr_scheduler)

    torch.save(model.state_dict(), "saves/supervised_model.pt")
    torch.save(optimiser.state_dict(), "saves/supervised_opt.pt")