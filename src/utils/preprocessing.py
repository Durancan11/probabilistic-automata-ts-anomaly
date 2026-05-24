import numpy as np

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def add_gaussian_noise(time_series, mean=0.0, std=0.05):
    noise = np.random.normal(mean, std, time_series.shape)
    return time_series + noise