import json
import numpy as np

class BaseModel:
    def __init__(self, input_shape) -> None:
        self.input_shape = input_shape

        self.W = np.random.randn(*((1,) + input_shape))
        self.b = np.random.randn(1)

        self.is_train = False

    def __call__(self, X):
        if X.shape[1:] != self.input_shape:
            raise f'Input shape wrong, expected {self.input_shape}'
        y_pre = X@self.W.T + self.b
        return y_pre.ravel()

    def parameters(self):
        return self.W, self.b

    def load(self, filename):
        with open(filename) as f:
            param = json.load(f)
        self.W, self.b = np.asarray(param['W']), np.asanyarray(param['b'])

    def save(self, filename):
        with open(filename, "w") as outfile:
            json.dump({'W': self.W.tolist(), 'b': self.b.tolist()}, outfile)

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

class LogisticRegression(BaseModel):
    def __init__(self, input_shape) -> None:
        super().__init__(input_shape)

    def __call__(self, X):
        return 1/(1 + np.exp(-super().__call__(X)))
