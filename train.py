from tqdm.auto import tqdm

from model import LogisticRegression
from loss import BinaryCrossEntropyLoss
from optimizer import GradientDescent, Momentum
from dataUtil import Dataset, DataLoader

def train(fn_train, fn_val, fo_weight, model, loss, optimizer, batch_size, epochs=10):
    dataset_train = Dataset(fn_train)
    train_loader = DataLoader(dataset_train, batch_size=batch_size)

    dataset_val = Dataset(fn_val)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)

    min_loss = float('inf')
    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    for _ in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_acc = 0
        for x_train, y_train in train_loader:
            train_loss += loss(model, x_train, y_train)
            y_hat = (model(x_train) >= 0.5).astype('int')
            train_acc += (y_hat == y_train).mean()
            optimizer.step()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        model.eval()
        val_loss = 0
        val_acc = 0
        for x_test, y_test in val_loader:
            val_loss += loss(model, x_test, y_test)
            y_hat = (model(x_test) >= 0.5).astype('int')
            val_acc += (y_hat == y_test).mean()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        if val_loss < min_loss:
            min_loss = val_loss
            model.save(fo_weight)

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

    return history

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Chooose option')
    parser.add_argument('-tr', '--fn_train', type=str, default="randomfaces4ar/dataset_train.mat")
    parser.add_argument('-te', '--fn_test', type=str, default="randomfaces4ar/dataset_test.mat")
    parser.add_argument('-m', '--model', type=str, default="logistic")
    parser.add_argument('-l', '--loss', type=str, default="b_entropy")
    parser.add_argument('-op', '--optimizer', type=str, default="momen")
    parser.add_argument('-oh', '--out_hist', type=str, default="hist.json")
    parser.add_argument('-ow', '--out_weight', type=str, default="weight.json")
    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-bz', '--batch_size', type=int, default=32)
    args = parser.parse_args()

    fn_train = args.fn_train
    fn_val = args.fn_test
    fo_weight = args.out_weight

    model = LogisticRegression((Dataset(fn_train)[0][0].shape[-1],))
    loss = BinaryCrossEntropyLoss()
    if args.optimizer == 'momen':
        optimizer = Momentum(model, loss)
    else:
        optimizer = GradientDescent(model, loss)

    hist = train(fn_train, fn_val, fo_weight, model, loss, optimizer, args.batch_size, args.epochs)

    import json
    with open(args.out_hist, "w") as outfile:
        json.dump(hist, outfile)

    # model = LogisticRegression((Dataset(fn_train)[0][0].shape[-1],))
    # model.load(fo_weight)
    # print(model.parameters())
