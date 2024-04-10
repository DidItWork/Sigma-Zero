import torch
from network import policyNN
from sim import generate_training_data
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from chess_tensor import actionsToTensor
import time
import torch.multiprocessing as mp

device = "cuda" if torch.cuda.is_available else "cpu"

class chessDataset(Dataset):

    def __init__(self, training_data):

        self.states = training_data["states"]
        self.actions = training_data["actions"]
        self.rewards = training_data["rewards"]

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):

        # actionTensor = actionsToTensor(self.actions[index], color=self.colours[index])[0]

        reward = torch.tensor(self.rewards[index], requires_grad = False)

        return self.states[index], self.actions[index], reward
    
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

def train(model=None, dataloader=None, optimiser=None) -> None:
    # shuffle(training_data)
    # loss = torch.zeros(1).to("cuda").requires_grad_(True)
    for idx, batch in enumerate(dataloader):
        # game_history = training_data[index]

        # for move in zip(game_history["states"], game_history["actions"], game_history["rewards"]):
        # print(move)
        # policy_mask = validActionsToTensor(move[1]).unsqueeze(0)

        # print(batch["states"])

        p, v = model(batch["states"].to(device))
        v = v.squeeze(-1)
        p_target = batch["actions"].to(device)
        v_target = batch["rewards"].to(device)
        # print(v.shape, v_target.shape)
        # p_target = torch.reshape(p_target, (p_target.size()[0], 1))

        # v_target = torch.tensor(v_target, dtype=torch.float32, device="cuda", requires_grad=True)
        # v = torch.tensor(v, dtype=torch.float32, device="cuda", requires_grad=True)

        # print(p, v)
        # print(p_target, v_target)
        # print(p.size(), p_target.size())
        # print(torch.log(p.to("cuda")).size())
        # print(torch.pow(torch.sub(v_target, v), 2).shape, torch.sum(torch.log(p)*p_target,dim=1).shape)
        loss = torch.sum(torch.sub(
                torch.pow(torch.sub(v_target, v), 2), 
                torch.sum(torch.log(p)*p_target,dim=1)
        ))

        print(f"iteration{idx}/{len(dataloader)} loss {loss}")

        
        # print(move_loss.size())
        # if torch.any(move_loss.isnan()):
        #     loss = torch.add(loss, move_loss)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


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
    model = policyNN(config=dict())
    # pretrained_weights = torch.load("test3.pt")
    # model.load_state_dict(pretrained_weights)
    
    generate_step = 10000
    num_steps = 200000
    num_games = 210
    num_process = 7
    args = {
        'C': 2,
        'num_searches': 50,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 64
    }
    
    # for batch in training_dataloader:
    #     print(batch)

    mp.set_start_method('spawn', force=True)

    optimiser = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, num_steps//generate_step*5)

    for epoch in range(num_steps//generate_step):

        print("Epoch", epoch)

        t1 = time.perf_counter()

        model = model.cpu()

        model.eval()

        manager = mp.Manager()
        return_dict = manager.dict()
            
        processes = []

        for i in range(num_process):

            processes.append(mp.Process(target = generate_training_data, args=(model, num_games//num_process, args, return_dict)))


        # training_data = generate_training_data(model, num_games, args)

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        

        training_data = {
            'states': [],
            'actions': [],
            'rewards': []
        }

        # print(return_dict)

        for game_dict in return_dict.values():

            for key in training_data:
                training_data[key] += game_dict[key]
        
        # print(training_data)

        training_dataset = chessDataset(training_data=training_data)

        training_dataloader = DataLoader(dataset=training_dataset,
                                        batch_size=args['batch_size'],
                                        shuffle=True,
                                        num_workers=4,
                                        collate_fn=training_dataset.collatefn,
                                        drop_last=True)

        model.train()
        model = model.to(device)

        for steps in range(generate_step//len(training_dataloader)):

            # print(model.conv1.weight)

            train(model=model, dataloader=training_dataloader, optimiser=optimiser)

            # print(f"Epoch {epoch} training complete")

        num_searches = args["num_searches"]
        batch_size = args["batch_size"]

        torch.save(model.state_dict(), f"./saves/train_{num_searches}_{batch_size}_{epoch}.pt")
        torch.save(optimiser.state_dict(), f"./saves/opt_{num_searches}_{batch_size}_{epoch}.pt")

        t2 = time.perf_counter()

        print(f"Time taken: {t2-t1:0.4f} seconds")

    print("Training complete")
    

if __name__ == "__main__":
    main()
