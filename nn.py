%pylab inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

Obs = np.loadtxt('Observations.csv', delimiter=',')

#Phase 1: Train one nn to get (x,y) on the past 10 angles

import csv
X = []
Y = []
Y_label = []

with open('Label.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        X.append([(Obs[int(row[0])-1][int(row[1])-10:int(row[1])])])
        Y.append([float(row[2]), float(row[3])])
        Y_label.append(row[2] + "," + row[3])

X_train = X[:540000]
X_validate = X[540000:]
Y_train = Y[:540000]
Y_validate = Y[540000:]
Y_label_validate = Y_label[540000:]
Y_label_validate = Y_label[540000:]

nsamples, nx, ny = np.shape(X_train)
X_train = np.reshape(X_train,(nsamples,nx*ny))

nsamples, nx, ny = np.shape(X_validate)
X_validate = np.reshape(X_validate,(nsamples,nx*ny))

from sklearn.neural_network import MLPRegressor
nn= MLPRegressor(solver='adam')
nn.fit(X_train,Y_train)

nn_ypred = nn.predict(X_validate)
mean_absolute_error(Y_validate,nn_ypred)

#Phase 2: Train another nn to map last 10 angles to the current angle. Then use the above nn to predict (x,y) for the 1001th angle
nsamples, nx, ny = np.shape(test_x)
test_x = np.reshape(test_x,(nsamples,nx*ny))

nn_obs = MLPRegressor(solver='adam')

test_obs= []
train_obs= []
obs_1000= []
for i in range(6000,10000):
    for j in range(10,len(Obs[i]),10):
        train_obs.append(Obs[i][j-10:j])
        test_obs.append(Obs[i][j])
    obs_1000.append(Obs[i][990:])
    
nn_obs.fit(train_obs,test_obs)

obs_1001 = nn_obs.predict(obs_1000)

obs_1001_v2 = []
ct=6000
for i in obs_1001:
    tmp = list()
    tmp= list(Obs[ct][991:])
    tmp.append(i)
    obs_1001_v2.append(list(tmp))
    ct+=1

prediction=nn.predict(obs_1001_v2)

#write final predictions
import csv

with open('nn_pair.csv', 'w') as myfile:
    wr = csv.writer(myfile,delimiter=',', quoting=csv.QUOTE_MINIMAL)
    wr.writerow(['id', 'value'])
    for idx, row in enumerate(prediction):
        row_id = str(6001+idx)
        wr.writerow([row_id+"x", row[0]])
        wr.writerow([row_id+"y", row[1]])