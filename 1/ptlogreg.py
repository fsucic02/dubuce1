import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import data

class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
        - D: dimensions of each datapoint 
        - C: number of classes
        """

        # inicijalizirati parametre (koristite nn.Parameter):
        # imena mogu biti self.W, self.b
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(C, D))
        self.b = torch.nn.Parameter(torch.zeros(C))

    def forward(self, X):
        # unaprijedni prolaz modela: izračunati vjerojatnosti
        # koristiti: torch.mm, torch.softmax
        scores = torch.mm(X, self.W.t()) + self.b
        return torch.softmax(scores, dim=1)

    def get_loss(self, X, Yoh_):
        # formulacija gubitka
        # koristiti: torch.log, torch.exp, torch.sum
        # pripaziti na numerički preljev i podljev
        scores = self.forward(X)
        logprobs = torch.log(scores)
        loss = -torch.sum(Yoh_ * logprobs) / X.shape[0]
        return loss

def train(model, X, Yoh_, param_niter, param_delta):
    """Arguments:
        - X: model inputs [NxD], type: torch.Tensor
        - Yoh_: ground truth [NxC], type: torch.Tensor
        - param_niter: number of training iterations
        - param_delta: learning rate
    """

    # inicijalizacija optimizatora
    optimizer = optim.SGD([model.W, model.b], lr=param_delta)

    # petlja učenja
    for i in range(param_niter):
        optimizer.zero_grad()
        loss = model.get_loss(X, Yoh_)
        loss.backward()
        optimizer.step()
        # ispisujte gubitak tijekom učenja
        if i % 10 == 0:
            print(f"iteration {i}: loss {loss.item()}")

def eval(model, X):
    """
    Arguments:
     - model: PyTorch model (PTLogreg)
     - X: actual datapoints [NxD], type: np.array
    Returns: predicted class probabilities [NxC], type: np.array
    """
    # ulaz je potrebno pretvoriti u torch.Tensor
    # izlaze je potrebno pretvoriti u numpy.array
    # koristite torch.Tensor.detach() i torch.Tensor.numpy()
    X_t = torch.Tensor(X)
    prediction = model.forward(X_t)
    return prediction.detach().numpy()

def logreg_decfun(model):
    def classify(X):
        return np.array([np.argmax(p) for p in eval(model, X)])

    return classify

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X, Y_ = data.sample_gauss_2d(3, 100)
    Yoh_ = data.class_to_onehot(Y_)
    X, Yoh_ = torch.from_numpy(X), torch.from_numpy(Yoh_)

    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])
    X = X.to(ptlr.W.dtype)

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, X, Yoh_, 5000, 0.5)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis = 1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, pr, M = data.eval_perf_multi(Y, Y_)
    AP = data.eval_AP(Y_)
    print(f"accuracy: {accuracy}, precision matrix: {pr}, confusion matrix: {M}, AP: {AP}")

    # iscrtaj rezultate, decizijsku plohu
    decfun = logreg_decfun(ptlr)
    rect = (torch.min(X, axis=0).values, torch.max(X, axis=0).values) 
    data.graph_surface(decfun, rect, offset=0)
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
