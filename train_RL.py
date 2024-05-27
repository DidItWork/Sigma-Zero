import torch
from network import policyNN
from sim import generate_training_data
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from chess_tensor import actionsToTensor
import time
import torch.multiprocessing as mp
from test_update import update_model
from lightning.pytorch.loggers import TensorBoardLogger

device = "cuda" if torch.cuda.is_available else "cpu"

class chessDataset(Dataset):

    def __init__(self, training_data):

        self.states = training_data["states"] #stored as boolean tensors
        self.actions = training_data["actions"] #stored as uci strings
        self.rewards = training_data["rewards"] #stored as integers
        self.colours = training_data["colours"] #stored as booleans, true for white
    
    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):

        action_tensor = actionsToTensor(self.actions[index], color=self.colours[index])[0]

        reward = torch.tensor(self.rewards[index], requires_grad = False, dtype=torch.float)

        return self.states[index], action_tensor, reward

    @staticmethod
    def collatefn(batch):

        indexes = torch.arange(8).view(1,1,8)

        states, actions, rewards = zip(*batch)

        actions = torch.stack(actions, dim=0)
        states = ((torch.stack(states, dim=0).to(dtype=torch.uint8).unsqueeze(-1) >> indexes)%2==1).to(dtype=torch.float)
        rewards = torch.stack(rewards, dim=0)

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }

@torch.no_grad()
def test(model=None, dataloader=None) -> dict:

    model.eval()

    mse_loss = torch.tensor(0.)
    ce_loss = torch.tensor(0.)

    for idx, batch in enumerate(dataloader):
        p, v = model(batch["states"].float().to(device))
        v = v.squeeze(-1)
        p_target = batch["actions"].to(device)
        v_target = batch["rewards"].to(device)

        mse_loss += torch.nn.functional.mse_loss(v, v_target).cpu()
        ce_loss += torch.nn.functional.cross_entropy(p, p_target).cpu()

        print(f"testing iteration {idx}/{len(dataloader)}")
    
    mse_loss /= len(dataloader)
    ce_loss /= len(dataloader)

    model.train()

    return {"mse_loss": mse_loss, "ce_loss": ce_loss}

def train(model=None,
          dataloader=None,
          test_dataloader=None,
          optimiser=None,
          total_steps=0,
          lr_scheduler=None,
          start_epoch=0,
          logger=None,
          cycle=0,
          supervised=True,
          test_step=1,
          log_step=10) -> None:

    test_dict = test(model, test_dataloader)
    best_loss = sum(list(test_dict.values()))

    for steps in range(start_epoch, total_steps+1):
        
        print(f"Step {steps}/{total_steps}")

        for pg in optimiser.param_groups:
            print("Learning Rate", pg["lr"])
            break
            
        model.train()

        for idx, batch in enumerate(dataloader):

            p, v = model(batch["states"].to(device))
            v = v.squeeze(-1)
            p_target = batch["actions"].to(device)
            v_target = batch["rewards"].to(device)

            mse_loss = torch.nn.functional.mse_loss(v, v_target)
            ce_loss = torch.nn.functional.cross_entropy(p, p_target)

            loss = mse_loss+ce_loss

            print(f"iteration {idx}/{len(dataloader)} ce_loss {ce_loss}, mse_loss {mse_loss}")

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
        
            if idx%log_step:
                logger.log_metrics({"MSE Loss":mse_loss, "CE Loss": ce_loss}, steps*len(dataloader)+idx)
        
        if steps%test_step==0 and test_dataloader is not None:
            test_dict = test(model, test_dataloader)
            test_loss = sum(list(test_dict.values()))
            print(f"Average test mse loss {test_dict['mse_loss']}")
            print(f"Average test ce loss {test_dict['ce_loss']}")
            print(f"Average test loss {test_loss}")
            print(f"Previous best test loss {best_loss}")
            if test_loss < best_loss:
                best_loss = test_loss
                if supervised:
                    torch.save(model.state_dict(), f"saves/supervised_model_best.pt")
                    torch.save(optimiser.state_dict(), f"saves/supervised_opt_best.pt")
                else:
                    torch.save(model.state_dict(), f"saves/RL_best.pt")
                    torch.save(optimiser.state_dict(), f"saves/RL_opt_best.pt")

        #Save every 5 steps
        if steps%5==0 and test_dataloader is None:
            if supervised:
                torch.save(model.state_dict(), f"saves/supervised_model_{steps}.pt")
                torch.save(optimiser.state_dict(), f"saves/supervised_opt_{steps}.pt")
    
    if supervised:
        torch.save(model.state_dict(), f"saves/supervised_model_{total_steps}.pt")
        torch.save(optimiser.state_dict(), f"saves/supervised_opt_{total_steps}.pt")
    else:
        torch.save(model.state_dict(), f"saves/RL_{cycle}.pt")
        torch.save(optimiser.state_dict(), f"saves/RL_opt_{cycle}.pt")

def main():

    model = policyNN(config=dict()).to(device)

    #Load Supervised Weights for self-play
    supervised_weights = torch.load("/home/benluo/school/Sigma-Zero/saves/supervised_model_15k_60.pt")
    
    model.load_state_dict(supervised_weights)

    num_games = 40
    num_process = 2

    args = {
        'C': 2,
        'num_searches': 100,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 30,
        'batch_size': 128,
        "start_epoch": 1,
        "chess960": True,
    }


    start_epoch = args["start_epoch"]
    num_epochs = args["num_epochs"]
    chess960 = args["chess960"]
    lr_step = 500

    mp.set_start_method('spawn', force=True)

    optimiser = Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    if start_epoch>1:
        try:
            pretrained_weights = torch.load(f"saves/RL_960_{start_epoch-1}.pt")
            optimiser_weights = torch.load(f"saves/RL_960_{start_epoch-1}.pt", map_location=device)
            model.load_state_dict(pretrained_weights)
            optimiser.load_state_dict(optimiser_weights)
        except:
            print(f"No saved weights from epoch {start_epoch-1} found!")
            start_epoch = 1
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=lr_step, gamma=0.95)
    
    manager = mp.Manager()

    logger = TensorBoardLogger("logs", name="RL_960")

    for epoch in range(start_epoch, num_epochs):

        print("Epoch", epoch)

        t1 = time.perf_counter()

        model = model.cpu()

        model.eval()

        return_dict = manager.dict()
            
        processes = []

        for i in range(num_process):

            processes.append(mp.Process(target = generate_training_data, args=(model, num_games//num_process, args, return_dict, chess960)))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        training_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'colours': [],
        }

        for game_dict in return_dict.values():

            for key in training_data:
                training_data[key] += game_dict[key]

        torch.save(training_data, f"games/RL_960_{epoch}.pt")
        
        del return_dict
        del processes

        training_dataset = chessDataset(training_data=training_data)

        training_dataloader = DataLoader(dataset=training_dataset,
                                        batch_size=args['batch_size'],
                                        shuffle=True,
                                        num_workers=1,
                                        collate_fn=training_dataset.collatefn,
                                        drop_last=True)

        model.train()
        model = model.to(device)

        train(model=model,
              dataloader=training_dataloader,
              optimiser=optimiser,
              total_steps=6,
              lr_scheduler=lr_scheduler,
              logger=logger,
              cycle=epoch)

        t2 = time.perf_counter()

        print(f"Time taken: {t2-t1:0.4f} seconds")

        #clear memory
        del training_dataloader
        del training_dataset
        del training_data

    print("Training complete")
    

if __name__ == "__main__":
    main()
