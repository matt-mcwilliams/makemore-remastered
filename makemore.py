

"""
A remaster of Andrej Karpathy's makemore!
"""



# -------- IMPORTS -----------

import os
import argparse
import math
import random
import time

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt


# ----------------------------
# Bigram Model
# ----------------------------

class Bigram():

    def __init__(self, generator=None):
        self.W = torch.randn((27,27), generator=generator, requires_grad=True)
    
    def forward(self, x):
        xenc = F.one_hot(x, num_classes=27).float()
        return xenc @ self.W
    
    def parameters(self):
        return [ self.W ]

    def context_length(self):
        return 1


# ----------------------------
# Trigram Model
# ----------------------------

class Trigram():

    def __init__(self, generator=None):
        self.W = torch.randn((27**2, 27), generator=generator, requires_grad=True)

    def forward(self, x):
        idx = x[...,0] + x[...,1] * 27
        return self.W[idx]
    
    def parameters(self):
        return [ self.W ]
    
    def weights(self):
        return [self.W]
    
    def context_length(self):
        return 2


# -------------------
# Multi-Activated Trigram (2 neurons activate every time)
# -------------------

class MultiActivatedTrigram():

    def __init__(self, generator):
        self.W = torch.randn((27*2,27), generator=generator, requires_grad=True)

    def forward(self, X):
        bit1 = F.one_hot(X[...,0], num_classes=27).float()
        bit2 = F.one_hot(X[...,1], num_classes=27).float()
        xenc = torch.cat((bit1, bit2), -1)
        return xenc @ self.W
    
    def parameters(self):
        return [ self.W ]
    
    def weights(self):
        return [self.W]
    
    def context_length(self):
        return 2


# -------------------------
# Multi Layer Perceptron
# -------------------------





class MultiLayerPerceptron():

    def __init__(self, context_length=7, hidden_neurons=200, feature_count=6, direct_connections=True, generator=None):
        self._context_length = context_length
        self._hidden_neurons = hidden_neurons
        self._feature_count = feature_count
        self._direct_connections = direct_connections

        self.C = torch.randn(27, feature_count, generator=generator, requires_grad=True)
        self.W1 = (torch.randn(feature_count*context_length, hidden_neurons, generator=generator) / math.sqrt(hidden_neurons)).clone().detach().requires_grad_(True)
        self.b1 = torch.zeros(hidden_neurons, requires_grad=True)
        self.W2 = (torch.randn(hidden_neurons, 27, generator=generator) / math.sqrt(27)).clone().detach().requires_grad_(True)
        self.b2 = torch.zeros(27, requires_grad=True)
        if direct_connections:
            self.WD = (torch.randn(feature_count*context_length, 27, generator=generator) / math.sqrt(27)).clone().detach().requires_grad_(True)
    
    def forward(self, X):
        xenc = self.C[X].view(-1, self._context_length*self._feature_count)
        h = torch.tanh(xenc @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        if self._direct_connections:
            logits += xenc @ self.WD
        return logits
    
    def parameters(self):
        return [ self.C, self.W1, self.b1, self.W2, self.b2 ] + ([self.WD] if self._direct_connections else [])
    
    def weights(self):
        return [ self.C, self.W1, self.W2 ] + ([self.WD] if self._direct_connections else [])
    
    def context_length(self):
        return self._context_length





# ==========================
# 
# TRAINING AND RUNNING
#
# ==========================



def train_model(model, X, Y, Xd=None, Yd=None, generator=None):

    # ---------- begin training ----------
    # TODO: periodically sample from the network
    
    epoch_count = 300000
    log_interval = 500
    initial_learning_rate = 0.1
    weight_decay_factor = 0.01

    start_time = time.time()
    losses = []
    for epoch in range(epoch_count):

        # create batch
        indicies = torch.randint(low=0, high=X.shape[0], size=(256,), generator=generator)
        Xb = X[indicies]
        Yb = Y[indicies]
        
        # forward pass
        logits = model.forward(Xb).view(-1,27)
        loss = F.cross_entropy(logits, Yb) + weight_decay_factor * sum([torch.mean(p**2) for p in model.weights()])

        losses.append(loss.log().item())
        if epoch % log_interval == 0:
            elapsed_time = time.time() - start_time
            eta = ((elapsed_time) / (epoch + 1)) * (epoch_count - epoch - 1)
            elapsed_minutes, elapsed_seconds = divmod(elapsed_time, 60)
            eta_minutes, eta_seconds = divmod(eta, 60)
            dev_loss = evaluate_loss(model, Xd if Xd is not None else Xb, Yd if Yd is not None else Yb)
            train_indices = torch.randint(low=0, high=X.shape[0], size=(5000,), generator=generator)
            train_loss = evaluate_loss(model, X[train_indices], Y[train_indices])
            print('=====================')
            print(f'Epoch {epoch} / {epoch_count}:')
            print(f'')
            print(f'Train Loss: {train_loss}')
            print(f'Dev Loss: {dev_loss}')
            print(f'Time: {int(elapsed_minutes)}m {int(elapsed_seconds)}s  |  ETA: {int(eta_minutes)}m {int(eta_seconds)}s')   
            print('=====================')    # backward pass
        for p in model.parameters():
            p.grad = None
        loss.backward()


        # update
        lr =  initial_learning_rate if epoch < epoch_count*0.6 else initial_learning_rate/10 # step learning rate decay
        for p in model.parameters():
            # print(p.shape)
            p.data += -lr * p.grad

    fig, ax = plt.subplots()
    ax.plot(list(range(epoch_count)), losses)
    plt.show()



def evaluate_loss(model, X, Y):
    # calculate loss
    logits = model.forward(X).view(-1,27)  
    loss = F.cross_entropy(logits, Y)

    return loss.item()



def load_model(model, model_name):
    weights = torch.load(f'models/{model_name}.pt')
    with torch.no_grad():
        for param, (name, weight) in zip(model.parameters(), weights.items()):
            param.copy_(weight)



def save_model(model, model_name):
    weights = { f'param_{i}':p for i,p in enumerate(model.parameters()) }
    torch.save(weights, f'models/{model_name}.pt')






if __name__ == '__main__':

    # ---------- arg parsing ----------

    parser = argparse.ArgumentParser(
        description="Description of your program",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--restart_training", "-r", action="store_true",
                        help="Restart the training of the currently selected model.")
    parser.add_argument("--model", "-m", default="trigram",
                        help="Model which you would use. Default is the current best model.")
    
    args = parser.parse_args()
    restart_training = args.restart_training
    model_choice = args.model


    # ==================================
    
    g = torch.Generator().manual_seed(314159265)

    # ---------- select model ----------

    match model_choice:
        
        case "bigram":
            model = Bigram(generator=g)
        
        case "trigram":
            model = Trigram(generator=g)
        
        case "ma-trigram":
            model = MultiActivatedTrigram(generator=g)

        case "mlp":
            model = MultiLayerPerceptron(generator=g)
        
        case _:
            raise Exception("Model is not availible!")


    # ---------- check for pre-trained network ----------
    
    pretrained = False
    if not restart_training and os.path.exists('models/bigram.pt'):
        load_model(model, model_choice)
        pretrained = True
    

    # ---------- load vocab -----------

    context_length = model.context_length()
    with open('names.txt', 'r') as file:

        words = ['.'*context_length + w.strip() + '.' for w in file.readlines()]

    vocab = sorted(list(set(''.join(words))))
    stoi = {c:i for i,c in enumerate(vocab)}
    itos = {i:c for c,i in stoi.items()}

    X = []
    Y = []

    for w in words:
        for i in range(len(w)-context_length-1):
            x = [ stoi[s] for s in w[i:i+context_length] ]
            y = stoi[ w[i+context_length+1] ]
            X.append(x)
            Y.append(y)
    
    # TODO: Refactor to use pytorch randomness to create consistancy.
    new_indicies = list(range(len(X)))
    random.shuffle(new_indicies)
    X = [X[i] for i in new_indicies]
    Y = [Y[i] for i in new_indicies]

    td_split = math.floor(0.8*len(X))
    dv_split = math.floor(0.9*len(X))

    Xt = torch.tensor(X[:td_split])
    Xd = torch.tensor(X[td_split:dv_split])
    Xv = torch.tensor(X[dv_split:])

    Yt = torch.tensor(Y[:td_split])
    Yd = torch.tensor(Y[td_split:dv_split])
    Yv = torch.tensor(Y[dv_split:])


    # ---------- train model -----------

    if not pretrained:
        train_model(model, Xt, Yt, Xd, Yd, generator=g)
        
        print('\n===================')
        print('Training complete!')
        print(f'Train set loss:  {evaluate_loss(model, Xt, Yt)}')
        print(f'Dev set loss:  {evaluate_loss(model, Xd, Yd)}')
        print('===================\n')

        save_model(model, model_choice)


    # -------- generate ten names -------------

    g = g.manual_seed(314159265)
    
    for i in range(10):

        context = [0] * context_length
        out = ''
        while True:
            ix = torch.tensor(context)
            logits = model.forward(ix)
            probs = F.softmax(logits, dim=1)

            generated_idx = torch.multinomial(probs, 1, generator=g, replacement=True).view(1)

            if generated_idx.item()==0:
                break

            out += itos[generated_idx.item()]
            context = context[1:] + [generated_idx.item()]
        
        print(out)
