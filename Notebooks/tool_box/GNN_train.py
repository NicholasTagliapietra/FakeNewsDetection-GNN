import torch
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
    
    for batch in data:
        batch.to(device)
        
        out = model(batch.x, batch.edge_index, batch.batch, batch.y[:, 1:])
        y_real = batch.y[:, 0].type(torch.int64)

        acc += get_acc(out, y_real)
        
    return round(acc/len(data), 2)


def train_step(model, data, optimizer, loss_f):
    model.train()
    tot_loss, acc = 0, 0
    for batch in data:
        batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch, batch.y[:, 1:])
        y_real = batch.y[:, 0].type(torch.int64)
        
        acc += get_acc(out, y_real)
        loss = loss_f(out, y_real) 
        tot_loss += loss.item()

        loss.backward()
        optimizer.step()

    return tot_loss/len(data), acc/len(data)
    #return round(tot_loss/len(data), 2), round(acc/len(data), 2)

@torch.no_grad()
def val_step(model, data, loss_f):
    model.eval()
    tot_loss, acc = 0, 0
    for batch in data:
        batch.to(device)
        
        out = model(batch.x, batch.edge_index, batch.batch, batch.y[:, 1:])
        y_real = batch.y[:, 0].type(torch.int64)

        acc += get_acc(out, y_real)
        tot_loss += loss_f(out, y_real).item()
    return tot_loss/len(data), acc/len(data)
    #return round(tot_loss/len(data), 2), round(acc/len(data), 2)

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

def train_all(model_class, datasets, emb_size=100, epochs=100, lr=0.001, wd=0.01, batch_size=128):
    models = []
    for x in datasets:
        print(f'--> Train on: {x[0]}')
        model = model_class(x[1].num_features, len(x[1].g_features), emb_size).to(device)
        hist = train(model, x[1], x[2], epochs=epochs, lr=lr, wd=wd, batch_size=batch_size)
        score = test_accuracy(model, x[3])
        models.append([x[0], model, hist, score])
    return models


def get_acc(y_pred, y_real):
    y_pred = y_pred.argmax(dim=-1)
    correct = int((y_pred == y_real).sum())
    return correct / len(y_real)


def plot_hist(hists):

    labels = ['Loss', 'Accuracy']
    accs, scores = [], {}

    for hist in hists:
        title, _, hist, score = hist
        scores[title] = score
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
    
    axs[1].set_title('Accuracy on test set')
    axs[1].set_xlabel('Datasets')
    axs[1].set_ylabel('Accuracy')
    axs[1].bar(scores.keys(), scores.values())
    plt.show()