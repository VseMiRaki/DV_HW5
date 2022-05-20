import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

class SOMNetwork():
    def __init__(self, dim, n, m, sigma=1, lr=1, iter=10000) -> None:

        self.dim = dim
        self.lr = lr
        self.init_lr = lr
        self.sigma = sigma
        self.shape = (m, n)
        self.max_iter=iter

        self.w = np.random.uniform(-1, 1, (n * m, dim))
        self.pos = np.array([np.where(np.full([m, n], True))[0], np.where(np.full([m, n], True))[1]]).T

        self.n = n
        self.m = m

        self.iter = 0
        self.momentum = None

    def competition(self, x):
        X = np.stack([x] * (self.m * self.n), axis=0)
        distance = np.linalg.norm(X - self.w, axis=1)
        return np.argmin(distance)

    def train(self, x):
        X = np.stack([x] * (self.n * self.m), axis=0)
        w_ind = self.competition(x)
        w_pos = self.pos[w_ind, :]

        W_pos = np.stack([w_pos] * (self.m * self.n), axis=0)
        distance = np.sum(np.square(self.pos.astype(np.float64) - W_pos.astype(np.float64)), axis=1)

        delta = np.exp((distance / (self.sigma ** 2)) * -1)
        step = delta * self.lr

        Step = np.stack([step]*(self.dim), axis=1)

        Delta = Step * (X - self.w)

        self.w += Delta

    def get_momentum(self, x):
        w_ind = self.competition(x)
        w = self.w[w_ind]
        return np.sum(np.square(x - w))
    
    def fit(self, X, epochs=1):
        size = X.shape[0]

        for epoch in range(epochs):
            print("Epoch", epoch)
            if self.iter > self.max_iter:
                print("Max iter reached")
                break

            indices = np.arange(size)
            for idx in indices:

                if self.iter > self.max_iter:
                    break

                input = X[idx]
                self.train(input)
                self.iter += 1
                self.lr = (1 - (self.iter / self.max_iter)) * self.init_lr

        self.momentum = np.sum(np.array([float(self.get_momentum(x)) for x in X]))    

    def transform(self, x):
        X = np.stack([x]*(self.m * self.n), axis=1)
        W = np.stack([self.w]*x.shape[0], axis=0)

        diff = X - W
        return np.linalg.norm(diff, axis=2)

    def predict(self, X):
        return np.array([self.competition(x) for x in X])

    def clusters(seld):
        return self.w.reshape(self.m, self.n, self.dim)


data_path = 'data.csv'
dataset = pd.read_csv(data_path)

data = dataset[dataset.columns.difference(['Id', 'Diagnosis'])]
labels = dataset['Diagnosis']
labels.replace(['B', 'M'], [0, 1], inplace=True)
data = data.to_numpy()


# Build a 2x1 SOM (2 clusters)
som = SOMNetwork(m=2, n=1, dim=30, lr=1, iter=100000, sigma=10)



som.fit(data, 50)

predictions = som.predict(data)

# Plot the results
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15,10))

# Extract just two features (just for ease of visualization)
x = data[:,3]
y = data[:,2]
colors = ['red', 'green']

print(predictions)

ax[0].scatter(x, y, c=labels.to_numpy(), cmap=ListedColormap(colors), marker='.')
ax[0].title.set_text('Actual Classes')
ax[1].scatter(x, y, c=predictions, cmap=ListedColormap(colors), marker='.')
ax[1].title.set_text('SOM Predictions')
plt.savefig('example.png')
plt.show()
