import torch
from network import policyNN
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from chess_tensor import actionsToTensor, tensorToAction
from train_RL import train, chessDataset, test
from lightning.pytorch.loggers import TensorBoardLogger

device = "cuda" if torch.cuda.is_available else "cpu"

def main(train_config=None, optimiser=None, lr_scheduler=None, logger=None, resume=False):

    start_epoch = train_config.get("start_epoch", 1)

    if start_epoch>1 or resume:

        try:

            optimiser_weights = torch.load(f"/home/benluo/school/Sigma-Zero/saves/supervised_opt_max_best_hlr.pt")
            model_weights = torch.load(f"/home/benluo/school/Sigma-Zero/saves/supervised_model_max_best_hlr.pt")

            optimiser.load_state_dict(optimiser_weights)
            model.load_state_dict(model_weights)
        
            print("Model and Optimizer weights loaded")

        except Exception as e:

            print(e)

            start_epoch = 1

    batch_size = train_config.get("batch_size", 128)
    num_epoch = train_config.get("epoch", 100)

    print("Loading dataset...")

    dataset = torch.load("game_data_60000.pt")

    print("Datasets loaded.")

    datasets = chessDataset(training_data=dataset)

    split_datasets = torch.utils.data.random_split(datasets, [0.8,0.2])

    train_dataloader = DataLoader(dataset=split_datasets[0],
                                  batch_size = batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=datasets.collatefn)

    test_dataloader = DataLoader(dataset=split_datasets[1],
                                  batch_size = batch_size,
                                  num_workers=0,
                                  collate_fn=datasets.collatefn)


    train(model=model,
          dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          optimiser=optimiser,
          total_steps=num_epoch,
          test_step=1,
          lr_scheduler=lr_scheduler,
          start_epoch=start_epoch,
          logger=logger)

if __name__ == "__main__":

    train_config = {
        "batch_size": 2048,
        "epoch": 20,
        "start_epoch": 1,
    }
    model_config = {
        "dropout": 0.,
        "value_dropout": 0.,
        "policy_dropout": 0.,
    }

    lr = 0.1
    lr_step_size = 4000
    weight_decay = 1e-4

    logger = TensorBoardLogger("logs", name="supervised_max")

    model = policyNN(model_config).to(device)

    optimiser = SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    lr_scheduler = StepLR(optimizer=optimiser, step_size=lr_step_size, gamma=0.5)

    main(train_config=train_config, optimiser=optimiser, lr_scheduler=lr_scheduler, logger=logger, resume=False)
