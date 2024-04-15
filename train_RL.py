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

        self.states = training_data["states"]
        self.actions = training_data["actions"]
        self.rewards = training_data["rewards"]
        self.colours = training_data["colours"]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):

        action_tensor = actionsToTensor(self.actions[index], color=self.colours[index])[0]

        action_tensor.requires_grad = False

        reward = torch.tensor(self.rewards[index], requires_grad = False, dtype=torch.float)

        return self.states[index], action_tensor, reward

    @staticmethod
    def collatefn(batch):

        states, actions, rewards = zip(*batch)

        # print(len(states), len(actions), len(rewards))

        states = torch.stack(states, dim=0)

        actions = torch.stack(actions, dim=0)

        rewards = torch.stack(rewards, dim=0)

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }

def train(model=None, dataloader=None, optimiser=None, total_steps=0, lr_scheduler=None, start_epoch=0) -> None:
    # shuffle(training_data)
    # loss = torch.zeros(1).to("cuda").requires_grad_(True)

    # base_model = policyNN({}).to(device)

    logger = TensorBoardLogger("logs", name="RL_960")

    for steps in range(start_epoch, total_steps):
        
        print(f"Step {steps}/{total_steps}")

        for pg in optimiser.param_groups:
            print("Learning Rate", pg["lr"])
            break
            
        model.train()

        for idx, batch in enumerate(dataloader):
            # game_history = training_data[index]

            # for move in zip(game_history["states"], game_history["actions"], game_history["rewards"]):
            # print(move)
            # policy_mask = validActionsToTensor(move[1]).unsqueeze(0)

            # print(batch["states"])

            p, v = model(batch["states"].float().to(device))
            v = v.squeeze(-1)
            p_target = batch["actions"].to(device)
            v_target = batch["rewards"].to(device)
            # print(p_target.dtype, v_target.dtype)
            # print(v, v_target)
            # print(p[0], p_target[0])
            # p_target = torch.reshape(p_target, (p_target.size()[0], 1))

            # v_target = torch.tensor(v_target, dtype=torch.float32, device="cuda", requires_grad=True)
            # v = torch.tensor(v, dtype=torch.float32, device="cuda", requires_grad=True)

            # print(p, v)
            # print(p_target, v_target)
            # print(p.shape, p_target.shape)
            # print(torch.log(p.to("cuda")).size())
            # print(torch.pow(torch.sub(v_target, v), 2).shape, torch.sum(torch.log(p)*p_target,dim=1).shape)
            # loss = torch.sum(torch.sub(
            #         torch.pow(torch.sub(v_target, v), 2), 
            #         torch.sum(torch.log(p)*p_target,dim=1)
            # ))

            mse_loss = torch.nn.functional.mse_loss(v, v_target)
            print("MSE Loss", mse_loss)
            ce_loss = torch.nn.functional.cross_entropy(p, p_target)

            loss = mse_loss+ce_loss

            print(f"iteration {idx}/{len(dataloader)} ce_loss {ce_loss}, mse_loss {mse_loss}")

            # print(move_loss.size())
            # if torch.any(move_loss.isnan()):
            #     loss = torch.add(loss, move_loss)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
        
            logger.log_metrics({"MSE Loss":mse_loss, "CE Loss": ce_loss})
        #Save at the end of epoch
        # with torch.inference_mode():

            # base_model.eval()
            # model.eval()

            # if update_model(current_model=base_model, new_model=model, matches=10):
                
                # base_model.load_state_dict(model.state_dict())
        if steps%5==0:

            torch.save(model.state_dict(), f"saves/RL_960_{steps}.pt")
            torch.save(optimiser.state_dict(), f"saves/RL_960_{steps}.pt")

                # print("new model saved!")


    # for game_history in training_data:
    #     # loss = torch.tensor()
    #     for move in zip(game_history["states"], game_history["actions"], game_history["rewards"]):
    #         p, v = self.forward(move[0].unsqueeze(0).cuda())
    #         c = 2
    #         print(p, v)
    #         # print(self.parameters)
    #         move_loss = torch.sub(
    #             torch.pow((move[2] - v), 2), 
    #             torch.add(
    #                 (move[1].T * torch.log(p)), 
    #                 (torch.mul(torch.pow(torch.abs(self.parameters), 2)), c)
    #             )
    #         )
    #         print(move_loss)
    #         print(move_loss.size())

def hw():
    print("Hello World!")

def main():
    model = policyNN(config=dict()).to(device)

    supervised_weights = torch.load("/home/benluo/school/Sigma-Zero/saves/supervised_model_15k_40.pt")
    
    model.load_state_dict(supervised_weights)

    generate_step = 1
    num_steps = 50
    num_games = 210 #210 games before epoch 6
    num_process = 7

    args = {
        'C': 2,
        'num_searches': 10,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 64,
        "start_epoch": 0
    }

    num_searches = args["num_searches"]
    batch_size = args["batch_size"]
    start_epoch = args["start_epoch"]
    
    # for batch in training_dataloader:
    #     print(batch)

    mp.set_start_method('spawn', force=True)

    optimiser = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    if start_epoch>0:
        try:
            pretrained_weights = torch.load(f"saves/train_{num_searches}_{batch_size}_{start_epoch-1}.pt")
            optimiser_weights = torch.load(f"saves/opt_{num_searches}_{batch_size}_{start_epoch-1}.pt", map_location=device)
            model.load_state_dict(pretrained_weights)
            optimiser.load_state_dict(optimiser_weights)
        except:
            print(f"No saved weights from epoch {start_epoch-1} found!")
            start_epoch = 0
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=int(0.9*generate_step), gamma=0.9)
    
    manager = mp.Manager()

    training_data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'colours': [],
    }

    for epoch in range(start_epoch, num_steps//generate_step):

        print("Epoch", epoch)

        t1 = time.perf_counter()

        model = model.cpu()

        model.eval()

        return_dict = manager.dict()
            
        processes = []

        for i in range(num_process):

            processes.append(mp.Process(target = generate_training_data, args=(model, num_games//num_process, args, return_dict, True)))

        # training_data = generate_training_data(model, num_games, args)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        # print(return_dict)

        for game_dict in return_dict.values():

            for key in training_data:
                training_data[key] += game_dict[key]

        torch.save(return_dict, f"games/RL_960_{epoch}.pt")
        
        del return_dict
        del processes
        
        # # print(training_data)

        # training_dataset = chessDataset(training_data=training_data)

        # training_dataloader = DataLoader(dataset=training_dataset,
        #                                 batch_size=args['batch_size'],
        #                                 shuffle=True,
        #                                 num_workers=1,
        #                                 collate_fn=training_dataset.collatefn,
        #                                 drop_last=True)

        # model.train()
        # model = model.to(device)

        # total_steps = generate_step//len(training_dataloader)

        

        # train(model=model, dataloader=training_dataloader, optimiser=optimiser, total_steps=total_steps, lr_scheduler=lr_scheduler)

        #     # print(f"Epoch {epoch} training complete")

        # # torch.save(model.state_dict(), f"./saves/train_{num_searches}_{batch_size}_{epoch}.pt")
        # # torch.save(optimiser.state_dict(), f"./saves/opt_{num_searches}_{batch_size}_{epoch}.pt")

        # t2 = time.perf_counter()

        # print(f"Time taken: {t2-t1:0.4f} seconds")

        # #clear memory
        # del training_dataloader
        # del training_dataset

    print("Training complete")
    

if __name__ == "__main__":
    main()
