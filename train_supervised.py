import torch
from network import policyNN
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from chess_tensor import actionsToTensor, tensorToAction
from train_RL import train, chessDataset

device = "cuda" if torch.cuda.is_available else "cpu"

def main(train_config=None, optimiser=None, lr_scheduler=None):

    start_epoch = train_config.get("start_epoch", 0)

    if start_epoch>0:

        try:

            optimiser_weights = torch.load(f"/home/benluo/school/Sigma-Zero/saves/supervised_opt_15k_{start_epoch-1}.pt")
            model_weights = torch.load(f"/home/benluo/school/Sigma-Zero/saves/supervised_model_15k_{start_epoch-1}.pt")

            optimiser.load_state_dict(optimiser_weights)
            model.load_state_dict(model_weights)
        
        except:

            start_epoch = 0

    batch_size = train_config.get("batch_size", 128)
    num_epoch = train_config.get("epoch", 100)

    dataset = torch.load("train_set_15k.pt")

    train_dataset = chessDataset(training_data=dataset)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size = batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  collate_fn=train_dataset.collatefn)

    train(model=model,
          dataloader=train_dataloader,
          optimiser=optimiser,
          total_steps=num_epoch,
          lr_scheduler=lr_scheduler,
          start_epoch=start_epoch)

if __name__ == "__main__":

    train_config = {
        "batch_size": 2048,
        "epoch": 61,
        "start_epoch": 56,
    }
    model_config = {
        "dropout": 0.,
        "value_dropout": 0.,
        "policy_dropout": 0.
    }

    lr = 0.005
    lr_step_size = 500
    weight_decay = 1e-4
    

    model = policyNN(model_config).to(device)

    model.train()

    optimiser = Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = StepLR(optimizer=optimiser, step_size=lr_step_size, gamma=0.95)

    main(train_config=train_config, optimiser=optimiser, lr_scheduler=lr_scheduler)
