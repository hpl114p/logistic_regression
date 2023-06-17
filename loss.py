import numpy as np

class BinaryCrossEntropyLoss:
    def __call__(self, model, X, y):
        y_hat = model(X)

        if model.is_train:
            self.dW = np.sum((y_hat - y)[:, None]*X, axis=0)/y.shape[0]
            self.db = np.sum((y_hat - y), axis=0)/y.shape[0]
        else:
            self.dW = 0
            self.db = 0

        loss = -y*np.log(y_hat) - (1-y)*np.log(1-y_hat+1e-15)
        return np.mean(loss, axis=0)
