import torch
import optuna
import numpy as np
import matplotlib.pyplot as plt


from tqdm.notebook import tqdm
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

@torch.no_grad()
def test_accuracy(model, data):    
    acc = 0
    model.eval()
    data = DataLoader(data, batch_size=128, shuffle=False)
    y_pred, y_real = [], []
    
    for batch in data:
        batch.to(device)
        
        out = model(batch.x, batch.edge_index, batch.batch, batch.y[:, 1:])
        y = batch.y[:, 0].type(torch.int64)
        
        y_real += list(y.detach().cpu().numpy())
        y_pred += list(out.detach().cpu().numpy())

        acc += get_acc(out, y)

    return acc/len(data), np.array(y_pred), np.array(y_real)


def train_step(model, data, optimizer, loss_f):
    model.train()
    tot_loss, acc = 0, 0
    
    g_feat = len(data.dataset.g_features) != 0
    
    for batch in data:
        batch.to(device)
        optimizer.zero_grad()
        
        if g_feat:
            out = model(batch.x, batch.edge_index, batch.batch, batch.y[:, 1:])
            y_real = batch.y[:, 0].type(torch.int64)
        else:
            out = model(batch.x, batch.edge_index, batch.batch, batch.y[:])
            y_real = batch.y[:].type(torch.int64)
            y_real = torch.squeeze(y_real)
            
        acc += get_acc(out, y_real)
        loss = loss_f(out, y_real) 
        tot_loss += loss.item()

        loss.backward()
        optimizer.step()

    return tot_loss/len(data), acc/len(data)
    

@torch.no_grad()
def val_step(model, data, loss_f):
    model.eval()
    tot_loss, acc = 0, 0
    
    n_feat = len(data.dataset.g_features) != 0
    
    for batch in data:
        batch.to(device)
        
        if n_feat:
            out = model(batch.x, batch.edge_index, batch.batch, batch.y[:, 1:])
            y_real = batch.y[:, 0].type(torch.int64)
        else:
            out = model(batch.x, batch.edge_index, batch.batch, batch.y[:])
            y_real = batch.y[:].type(torch.int64)
            y_real = torch.squeeze(y_real)
                                                                        
        acc += get_acc(out, y_real)
        tot_loss += loss_f(out, y_real).item()
    return tot_loss/len(data), acc/len(data)
    

def train(model, data_train, data_val, epochs=100, lr=0.001, wd=0.01, batch_size=128):
    acc_losses_t, acc_losses_v = [], []
    loss_f = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(epochs)):
        loss_t, acc_t = train_step(model, train_loader, optimizer, loss_f)
        loss_v, acc_v = val_step(model, val_loader, loss_f)
        
        acc_losses_t.append([loss_t, acc_t])
        acc_losses_v.append([loss_v, acc_v])
        if (epoch+1) % 10 == 0:
            print(f'Epochs: {epoch+1} | loss_train={loss_t}  loss_val={loss_v} | acc_train={acc_t}  acc_val={acc_v}')
    return acc_losses_t, acc_losses_v



def train_all_and_optimize(model_class, datasets, epochs=100):
    models = []
    
    for x in datasets:
        print(f'--> Train on: {x[0]}')
        embedding_space_dim, learning_rate, weight_decay, batch_size = GridSearch(model_class, x[1], 
                                                                                  x[2], num_trials = 40, 
                                                                                  epochs_max = epochs)
        # Train Model with the Best Hyperparameters
        # Initialize Model with best Hyperparameters
        num_node_features = x[1].num_features
        num_graph_features = len(x[1].g_features)
        best_model = model_class(num_n_feature = num_node_features, 
                                 num_g_feature = num_graph_features, 
                                 emb_size = embedding_space_dim).to(device)
        hist = train(best_model, x[1], x[2], epochs=epochs, 
                     lr=learning_rate, wd=weight_decay, 
                     batch_size=batch_size)
        score, _, _ = test_accuracy(best_model, x[3])
        models.append([x[0], best_model, hist, score])
        
    return models

def GridSearch(GNN, train_dataset, val_dataset, num_trials = 40, epochs_max = 60):
    search_space = {"learning_rate":[0.001, 0.005, 0.01],
                    "weight_decay":[0.001, 0.005, 0.01],
                    "batch_size":[128, 256, 512],
                    "embedding_space_dim":[60,80,100,120,140,160]}
    study = optuna.create_study(direction = "maximize", sampler = optuna.samplers.GridSampler(search_space))
    objective = Objective(GNN, train_dataset,val_dataset, epochs_max = epochs_max)
    study.optimize(objective, n_trials = num_trials)
    
    # Extract the best hyperparameters
    embedding_space_dim = study.best_params["embedding_space_dim"]
    learning_rate = study.best_params["learning_rate"]
    weight_decay = study.best_params["weight_decay"]
    batch_size = study.best_params["batch_size"]
    
    return embedding_space_dim, learning_rate, weight_decay, batch_size

class Objective(object):
    
    def __init__(self,GNN, train_dataset, val_dataset, epochs_max = 60):
        self.GNN = GNN
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs_max = epochs_max
        
    def __call__(self,trial):
        
        # Generate Set of Hyperparameters to Test
        learning_rate = trial.suggest_categorical("learning_rate",[0.001, 0.005, 0.01])
        weight_decay = trial.suggest_categorical("weight_decay",[0.001, 0.005, 0.01])
        batch_size = trial.suggest_categorical("batch_size",[64, 128, 256, 512])
        embedding_space_dim = trial.suggest_categorical("embedding_space_dim",[40,60,80,100,120,140,160])


        # If the trial has already been explored, prune it. It may happen because 
        # the hyperparameter optimizer searchs near the most promising values.
        for t in trial.study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.params == trial.params:
                raise optuna.exceptions.TrialPruned('Duplicate Parameter Set')
        
        # Generate Model
        num_node_features = self.train_dataset.num_features
        num_graph_features = len(self.train_dataset.g_features)
        model = self.GNN(num_n_feature = num_node_features, 
                         num_g_feature = num_graph_features, 
                         emb_size = embedding_space_dim).to(device)

        # Initialize Dataloaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)


        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr = learning_rate, 
                                     weight_decay = weight_decay)
        loss_f = torch.nn.NLLLoss()

        return train_optuna(trial, model, optimizer, loss_f, 
                            train_loader, val_loader, 
                            epochs = self.epochs_max)
    
    
def train_optuna(trial, model, optimizer, loss_f, train_loader, val_loader, epochs = 60):
    
  acc_losses_t, acc_losses_v = [], []
  acc_v = 999
  for epoch in range(epochs):
    loss_t, acc_t = train_step(model, train_loader, optimizer, loss_f)
    loss_v, acc_v = val_step(model, val_loader, loss_f)
        
    acc_losses_t.append([loss_t, acc_t])
    acc_losses_v.append([loss_v, acc_v])

    trial.report(acc_v, epoch)

    # Early Stopping
    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()

  return acc_v




def get_acc(y_pred, y_real):
    y_pred = y_pred.argmax(dim=-1)
    correct = int((y_pred == y_real).sum())
    return correct / len(y_real)


def plot_hist(hists):

    labels = ['Loss', 'Accuracy']
    accs, scores, titles = [], [], []
    
    for hist in hists:
        title, _, hist, score = hist
        scores.append(score)
        titles.append(title)
        acc_losses_t, acc_losses_v = np.asarray(hist[0]), np.asarray(hist[1])
        accs.append([title, acc_losses_v[:, 1]])
        fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
        fig.suptitle(f'Dataset: {title}', fontsize=16)
        for i, l in enumerate(labels):
            axs[i].plot(acc_losses_t[:, i], label=f'{l} train')
            axs[i].plot(acc_losses_v[:, i], '--', label=f'{l} val')
            axs[i].set_title(f'{l} evolution')
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel(l)
            axs[i].grid()
            axs[i].legend()
        plt.show()

    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 5))
    fig.suptitle('Resume', fontsize=16)
    
    for x in accs:
        axs[0].set_title('Evolution of val-accuracy')
        axs[0].plot(x[1], label=x[0])
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].grid()
    
    round_scores = [round(score, 3) for score in scores]
    axs[1].set_title(f'Accuracy on test set: {round_scores}')
    axs[1].set_xlabel('Datasets')
    axs[1].set_ylabel('Accuracy')
    axs[1].bar(titles, scores)
    plt.show()
    


  