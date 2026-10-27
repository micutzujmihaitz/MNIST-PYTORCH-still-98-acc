#in mod evident aici import toate librariile necesare

import numpy as np
import torch  
import torch.nn as nn  
from torch.utils.data import TensorDataset, DataLoader  
from sklearn.datasets import fetch_openml
X,y = fetch_openml("mnist_784" , version=1 , return_X_y=True, as_frame=False, parser="liac-arff")
print(X.shape)
print(y.shape)
#astype prelucreaza datele astfel incat sa poata fi folosite de pytorch si pe x il imparte la 255
#si torch from numpy le transforma in matrici 
X = X.astype(np.float32)/255
y = y.astype(int)

X = torch.from_numpy(X)
y = torch.from_numpy(y)


#aici este impartirea ca la modelul trecut 
X_antrenare = X[:60000]
X_testare = X[60001:]
Y_antrenare = y[:60000]
Y_testare = y[60001:]


#TensorDataset ii atribuie lui X_antrenare[0] valoarea lui y_antrenare[0] 
#si dataloader e pur si simplu bucla for pe care o foloseam in codul facut cu numpy

train_dataset = TensorDataset(X_antrenare,Y_antrenare)
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

class Retea(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,10)
        __init__ : nn.init.kaiming_normal_(self.fc1.weight,nonlinearity="relu")
        __init__ : nn.init.kaiming_normal_(self.fc1.bias) 
        __init__ : nn.init.kaiming_normal_(self.fc2.weight,nonlinearity="relu")
        __init__ : nn.init.kaiming_normal_(self.fc2.bias)    
    def forward(self,x):
        z1 = self.fc1(x)
        a1 = torch.relu(z1)
        return self.fc2(a1)

model = Retea()
L = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

for epoca in range (10) : 
    sumawrong = 0
    for Xbatch,Ybatch in train_loader :
        optimizer.zero_grad() # de retinut aceasta functie este foarte importanta pentru ca lasa algoritmul sa aiba un gradient unic de ficare data
        z2 = model(X_antrenare) # asta e practic forward passul
        loss = L(z2,Y_antrenare)
        sumawrong += loss.item()*Xbatch.size(0)
        loss.backward()
        optimizer.step() #aici se intampla toaata magia si anume se modifica ponderile
    print (sumawrong/len(train_dataset))

torch.save(model.state_dict(), "mnist_mlp.pt")

np.save("Xtotal",X)
np.save("Ytotal",y)