import numpy as np
import torch  
import torch.nn as nn  
from torch.utils.data import TensorDataset, DataLoader  
from retea import Retea #Nu stiam ca poti sa faci asta dar aparent poti 

Xtotal = np.load("Xtotal.npy")
Ytotal = np.load("Ytotal.npy")

X = Xtotal.astype(np.float32)
y = Ytotal.astype(int)

X = torch.from_numpy(X)
y = torch.from_numpy(y)

X_testare = X[60001:]
Y_testare = y[60001:]


model = Retea()
#model.load_state_dict(torch.load("mnist_mlp.pt"))

model.eval()

total = 0

with torch.no_grad():
    #aici incarc ponderile pe care le am facut cu numpy
    weight1 = np.load("W1 (1).npy")
    weight2 = np.load("W2.npy")
    bias1 = np.load("b1.npy")
    bias2 = np.load("b2.npy")
    #aici le transform in tensori
    weight1 = torch.from_numpy(weight1)
    weight2 = torch.from_numpy(weight2)
    bias1= torch.from_numpy(bias1)
    bias2 = torch.from_numpy(bias2)
    #abia de acum le pot folosi cum trebiue
    model.fc1.weight.copy_(weight1)
    model.fc2.weight.copy_(weight2)
    model.fc1.bias.copy_(bias1)
    model.fc2.bias.copy_(bias2)

    z2 = model(X_testare)
    pred = z2.argmax(dim=1)
    acuratete = (pred == Y_testare).float().mean()

print (acuratete.item() * 100)

