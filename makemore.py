

"""
A remaster of Andrej Karpathy's makemore!
"""



# -------- IMPORTS -----------

import torch
import torch.nn.functional as F
import os
import argparse


# ----------------------------
# Bigram Model
# ----------------------------

class Bigram():

    def __init__(self, vocab_size, generator=None):
        self.W = torch.randn((27,27), generator=generator, requires_grad=True)
    
    def forward(self, x):
        xenc = F.one_hot(x, num_classes=27).float()
        return xenc @ self.W
    
    def parameters(self):
        return [ self.W ]

    def context_length(self):
        return 1
    



def train_model(model):
    context_length = model.context_length()
    with open('names.txt', 'r') as file:

        words = ['.'*context_length + w.strip() + '.' for w in file.readlines()]

    vocab = sorted(list(set(''.join(words))))
    stoi = {c:i for i,c in enumerate(vocab)}
    itos = {i:c for c,i in stoi.items()}

    xs = []
    ys = []

    for w in words:
        for i in range(len(w)-context_length-1):
            x = [ stoi[s] for s in w[i:i+context_length] ]
            y = stoi[ w[i+context_length+1] ]
            xs.append(x)
            ys.append(y)


    
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)


    # ---------- begin training ----------
    # TODO: periodically sample from the network

    for epoch in range(200):
        
        # forward pass
        logits = model.forward(xs).view(-1,27)
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(probs.size(0)), ys].log().mean()

        print(f'{epoch}, {loss.item()}')


        # backward pass
        for p in model.parameters():
            p.grad = None
        loss.backward()


        # update
        for p in model.parameters():
            p.data += -10 * p.grad



def save_model(model):
    torch.save(model.W, 'models/bigram.pt')





if __name__ == '__main__':

    # ---------- arg parsing ----------

    parser = argparse.ArgumentParser(
        description="Description of your program",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--restart_training", "-r", action="store_true",
                        help="Restart the training of the currently selected model.")
    
    restart_training = parser.parse_args().restart_training


    # ==================================
    
    g = torch.Generator().manual_seed(314159265)

    # ---------- select model ----------

    model = Bigram(27, generator=g)


    # ---------- check for pre-trained network ----------
    
    pretrained = False
    if not restart_training and os.path.exists('models/bigram.pt'):
        model.W = torch.load('models/bigram.pt')
        pretrained = True


    # ---------- train model -----------

    if not pretrained:
        train_model(model)
        save_model(model)
    

    # ---------- load vocab -----------

    context_length = model.context_length()
    with open('names.txt', 'r') as file:

        words = ['.'*context_length + w.strip() + '.' for w in file.readlines()]

    vocab = sorted(list(set(''.join(words))))
    stoi = {c:i for i,c in enumerate(vocab)}
    itos = {i:c for c,i in stoi.items()}


    # -------- generate ten names -------------
    
    for i in range(10):

        ix = torch.tensor(0)
        out = ''
        while True:
            logits = model.forward(ix)
            counts = logits.exp()
            probs = counts / counts.sum()

            ix = torch.multinomial(probs, 1, replacement=True).view(1)

            if ix.item()==0:
                break

            out += itos[ix.item()]
        
        print(out)
