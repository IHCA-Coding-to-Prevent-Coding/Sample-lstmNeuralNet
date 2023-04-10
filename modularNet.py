from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import SGD as SGD
import pandas as pd
from torch import tensor
from torch import no_grad
from statistics import mean
from sklearn.metrics import roc_auc_score

# model hyperparamaters
inputSize=5 # variables input
hiddenSizeOne=32 # nodes of hidden layer
hiddenSizeTwo=32 # nodes of hidden layer
outSize=1 # power consumptoion in zone 1
batchSize= 144*7 # how many data points are in batch
lr = 0.025
seqLength=144 # num of time values per input
epochSize = 10
N_SPLITS = 7 # num of training/validation splits

# helper
class extractTensor(nn.Module):
    def forward(self, x):
        tensor, hs = x
        return tensor.reshape(-1, hiddenSizeOne) 
    
# model itself
model = nn.Sequential(
    nn.LSTM(inputSize, hiddenSizeOne),
    extractTensor(),
    nn.Sigmoid(),
    nn.Linear(hiddenSizeOne, hiddenSizeTwo),
    #nn.Sigmoid(),
    nn.Linear(hiddenSizeTwo, outSize)
    #nn.Sigmoid() 
    #nn.Linear(linearSize, outSize)
)

#getting weather data
data = pd.read_csv('Tetuan City power consumption.csv')
data = data.iloc[:5000, :]

X = data.iloc[:, [1, 2, 3, 4, 5]]
Y = data.iloc[:, [6]]
assert len(X) == len(Y)

# TODO train_test_split for validation / everything else
#X, Xtest, Y,Ytest = train_test_split(X, Y, train_size=0.8)

skf = StratifiedKFold(n_splits=N_SPLITS)
MSEloss = nn.MSELoss()
optimizer = SGD(model.parameters(), lr=lr)

# convering Y into binary
Y = tensor(Y.values).float()
average = mean([i.item() for i in Y])
Y = [float(i.item()>average) for i in Y]

# K-fold splitting data into training and testing
allLosses = []
for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
    
    y = [Y[i] for i in train_index]

    x = pd.DataFrame()
    x['Temperature'] = [X['Temperature'][i] for i in train_index]
    x['Humidity'] = [X['Humidity'][i] for i in train_index]
    x['Wind Speed'] = [X['Wind Speed'][i] for i in train_index]
    x['general diffuse flows'] = [X['general diffuse flows'][i] for i in train_index]
    x['diffuse flows'] = [X['diffuse flows'][i] for i in train_index]

    x = tensor(x.values).float()
    y = tensor(y).float()

    mselist = []
    for j in range(epochSize):
        pred = model(x).squeeze(1) # seems to increase auroc (SIGNIFICANTLY ~30-60%), don't know why, doesn't remove siez difference warning

        mse = MSEloss(pred, y)
        mselist.append(mse.item())

        optimizer.zero_grad()
        mse.backward()
        optimizer.step()

    with no_grad():

        xtest = pd.DataFrame()
        xtest['Temperature'] = [X['Temperature'][i] for i in test_index]
        xtest['Humidity'] = [X['Humidity'][i] for i in test_index]
        xtest['Wind Speed'] = [X['Wind Speed'][i] for i in test_index]
        xtest['general diffuse flows'] = [X['general diffuse flows'][i] for i in test_index]
        xtest['diffuse flows'] = [X['diffuse flows'][i] for i in test_index]
        xtest = tensor(xtest.values).float()

        y = [Y[i] for i in test_index]
        y = tensor(y).float()

        pred = model(xtest).squeeze(1)

        valmse = MSEloss(pred, y)
        aucscores = []
        aucscores.append(roc_auc_score(y, pred))
        print(f'Fold {i}, Total training MSE change: {mselist[0]-mselist[-1]}')
        print(f'Fold {i}, Validation MSE / Total training MSE * 100: {mselist[-1]/valmse*100}')
        print(f'Fold {i}, AUROC score: {aucscores[-1]}')
        print('#################')

scoremean = mean(aucscores)
print(f'Average auroc score: {scoremean}')
with open("Performance.txt", "a") as file:
    file.writelines(f'Average Score: {scoremean}\n{model}\n####################\n')
# TODO split validation data