import os 
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn1 = nn.GRU(input_size=input_size, hidden_size=hidden_size,num_layers=4)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
#        x = np.array(x)
#        x = x.reshape([1,1,17])
#        x = torch.reshape(input=x,shape=(1,1,17))
#        x = x.resize_((1,1,17))
        x = self.rnn1(input=x)
        print('asa',x)
        x = x[0]
        x = F.relu(x)
        x = self.linear2(input=x)
#        x = torch.reshape(x,(1,3))
        print(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # (n, x)
        print(state.shape)

        if len(action.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
#            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        next_state = next_state.resize_((len(next_state),1,1,11)) 
        print(state.shape)


        # 1: predicted Q values with current state
#        target = torch.empty(len(state),3)
#        for idx in range(len(done)):
#            pred = self.model(state[idx])
#            torch.add(target, pred) 
#        print(target.shape)
        
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new
        
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
        
