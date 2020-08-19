from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

batch_size = 32

train_set = torch.Tensor(train_X.values, dtype=torch.float32)
train_loader = DataLoder(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
test_set = torch.Tensor(test_X.values, dtype=torch.float32)
test_loader = DataLoder(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

class Net(nn.Module):
    def __init__(self, activation):
        super(Net, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 10, kernel_size=3)

    def forward(self, x):
        x = self.activation(F.max_pool2d(self.conv1(x), 2))
        x = self.activation(F.max_pool2d(self.conv2(x), 2))
        x = self.activation(F.max_pool2d(self.conv3(x), 2))
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        return x.view(-1, 10)

        def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for data, target in train_loader:
      data, target = data.to(device), target.to(device)
      
      # Zero the parameter gradients
      optimizer.zero_grad()
      
      # forward + backward + optimize
      output = model(data)
      
      loss = criterion(output, target)
      loss.backward()
      
      optimizer.step()
      

def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
    accuracy = 1 - correct / len(test_loader.dataset)
    return accuracy

    # 1. Categorical parameter
optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])

# 2. Int parameter
num_layers = trial.suggest_int('num_layers', 1, 3)

# 3. Uniform parameter
dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)

# 4. Loguniform parameter
learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

# 5. Discrete-uniform parameter
drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1)

import torch.optim as optim

def get_optimizer(trial, model):
  # optimizer をAdamとMomentum SGDで探索
  optimizer_names = ['Adam', 'MomentumSGD']
  optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)
    
  # weight decayの探索
  weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
  
  # optimizer_nameで分岐
  if optimizer_name == optimizer_names[0]: 
      adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
      optimizer = optim.Adam(model.parameters(), lr=adam_lr, weight_decay=weight_decay)
  else:
      momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
      optimizer = optim.SGD(model.parameters(), lr=momentum_sgd_lr,
                            momentum=0.9, weight_decay=weight_decay)
  return optimizer


def get_activation(trial):
  # 活性化関数の探索. ReLU or ELu.
    return trial.suggest_categorical('activation', [F.relu, F.elu])


import optuna
import torch

EPOCH = 5

def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    activation = get_activation(trial)

    model = Net(activation).to(device)
    optimizer = get_optimizer(trial, model)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    for step in range(EPOCH):
        train(model, device, train_loader, optimizer, criterion)
        
    # Evaluation
    accuracy = evaluate(model, device, test_loader)
  
    # 返り値が最小となるようにハイパーパラメータチューニングが実行される
    return 1.0 - accuracy


study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params  # Get best parameters for the objective function.
study.best_value  # Get best objective value.
study.best_trial  # Get best trial's information.
study.trials  # Get all trials' information.
study.trials_dataframe()  # Get a pandas dataframe like