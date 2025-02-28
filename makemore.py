

"""
A remaster of Andrej Karpathy's makemore!
"""



# -------- IMPORTS -----------

import os
import argparse
import math
import random

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
        idx = x[...,0] + x[...,1] * 27   # [ examples x 2 (context) ]
        # xenc = F.one_hot(idx, num_classes=27**2).float()  # [ examples ]
        # return xenc @ self.W
        return self.W[idx]
    
    def parameters(self):
        return [ self.W ]
    
    def context_length(self):
        return 2



# ==========================




def train_model(model, X, Y, generator=None):

    # ---------- begin training ----------
    # TODO: periodically sample from the network
    
    epoch_count = 500
    losses = []
    for epoch in range(epoch_count):

        # create batch
        indicies = torch.randint(low=0, high=X.shape[0], size=(8192,), generator=generator)
        Xb = X[indicies]
        Yb = Y[indicies]
        
        # forward pass
        logits = model.forward(Xb).view(-1,27)
        loss = F.cross_entropy(logits, Yb)

        losses.append(loss.item())
        if epoch % 20 == 0:
            print(f'{epoch=}: {loss.item()=}')


        # backward pass
        for p in model.parameters():
            p.grad = None
        loss.backward()


        # update
        lr = 10**(-epoch/epoch_count) * 100 # 100 to 10, logarithmic scale
        for p in model.parameters():
            p.data += -lr * p.grad

    fig, ax = plt.subplots()
    ax.plot(list(range(epoch_count)), losses)
    ax.plot(list(range(epoch_count)), [2.57]*epoch_count)
    plt.show()


def evaluate_loss(model, X, Y):
    # calculate loss
    logits = model.forward(X).view(-1,27)  
    loss = F.cross_entropy(logits, Y)

    return loss.item()



def load_model(model, model_name):
    model.W = torch.load(f'models/{model_name}.pt')



def save_model(model, model_name):
    torch.save(model.W, f'models/{model_name}.pt')





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
        train_model(model, Xt, Yt, generator=g)
        
        print('\n===================')
        print('Training complete!')
        print(f'Train set loss:  {evaluate_loss(model, Xt, Yt)}')
        print(f'Dev set loss:  {evaluate_loss(model, Xd, Yd)}')
        print('===================\n')

        save_model(model, model_choice)


    # -------- generate ten names -------------

    g = g.manual_seed(314159265)
    
    for i in range(10):

        ix = torch.tensor([0]*context_length)
        out = ''
        while True:
            logits = model.forward(ix)
            counts = logits.exp()
            probs = counts / counts.sum()

            generated_idx = torch.multinomial(probs, 1, replacement=True).view(1)

            if generated_idx.item()==0:
                break

            out += itos[generated_idx.item()]
            ix[:-1] = ix[1:]
            ix[-1] = generated_idx
        
        print(out)
