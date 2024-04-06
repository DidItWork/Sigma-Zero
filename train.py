import torch
from network import policyNN
from sim import generate_training_data
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from chess_tensor import actionsToTensor

class chessDataset(Dataset):

    def __init__(self, training_data):

        self.states = training_data["states"]
        self.actions = training_data["actions"]
        self.rewards = training_data["rewards"]
        self.colours = training_data["colours"]

    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, index):

        actionTensor = actionsToTensor(self.actions[index], color=self.colours[index])[0]

        actionTensor.requires_grad = False

        reward = torch.tensor(self.rewards[index], requires_grad = False)

        return self.states[index], actionTensor, reward
    
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
        p, v = model(batch["states"].cuda())
        v = v.squeeze(-1)
        p_target = batch["actions"].cuda()
        v_target = batch["rewards"].cuda()

        # print(p.shape, p_target.shape)
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = policyNN(config=dict()).to(device)
    optimiser = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    model.train()
    num_epochs = 500
    args = {
        'C': 2,
        'num_searches': 10,
        'num_iterations': 3,
        'num_selfPlay_iterations': 500,
        'num_epochs': 4,
        'batch_size': 16
    }
    
    # for batch in training_dataloader:
    #     print(batch)

    training_data = generate_training_data(model, num_games=1, args=args)

    training_dataset = chessDataset(training_data=training_data)

    training_dataloader = DataLoader(dataset=training_dataset,
                                    batch_size=args['batch_size'],
                                    shuffle=True,
                                    num_workers=2,
                                    collate_fn=training_dataset.collatefn,
                                    drop_last=True)

    for epoch in range(num_epochs):

        print(f"Epopch {epoch}")

        train(model=model, dataloader=training_dataloader, optimiser=optimiser)

        # print(f"Epoch {epoch} training complete")

    torch.save(model.state_dict(), "./test3.pt")
    torch.save(optimiser.state_dict(), "./opt3.pt")
    print("Training complete")


if __name__ == "__main__":
    main()
