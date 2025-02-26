

"""
A remaster of Andrej Karpathy's makemore!
"""



# -------- IMPORTS -----------

import torch
import torch.nn.functional as F
import os


# ----------------------------
# Bigram Model
# ----------------------------

class Bigram():

    def __init__(self, vocab_size, generator=None):
        self.W = torch.randn((27,27), generator=generator, requires_grad=True)
    
    def forward(self, xenc):
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

    # TODO: Allow for longer context lengths. currently this approach only works for cl = 1
    for w in words:
        pairs = [(x,y) for x,y in zip(w, w[1:])]
        indexed_pairs = [(stoi[x], stoi[y]) for x,y in pairs]
        xs += [x for x,y in indexed_pairs]
        ys += [y for x,y in indexed_pairs]
    
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)


    # ---------- begin training ----------
    # TODO: periodically sample from the network

    for epoch in range(200):
        
        # forward pass
        xenc = F.one_hot(xs, num_classes=27).float()
        logits = model.forward(xenc)
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(probs.size(0)), ys].log().mean()

        print(loss)


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
    
    g = torch.Generator().manual_seed(314159265)

    # ---------- select model ----------

    model = Bigram(27, generator=g)


    # ---------- check for pre-trained network ----------
    
    pretrained = False
    if os.path.exists('models/bigram.pt'):
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


    # generate ten names
    
    for i in range(10):

        ix = torch.tensor(0) # []
        out = ''
        while True:
            xenc = F.one_hot(ix, num_classes=27).float()  # [27]
            logits = model.forward(xenc)
            counts = logits.exp()
            probs = counts / counts.sum()

            ix = torch.multinomial(probs, 1, replacement=True).view(1)

            if ix.item()==0:
                break

            out += itos[ix.item()]
        
        print(out)
