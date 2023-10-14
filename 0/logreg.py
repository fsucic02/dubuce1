import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import data
import matplotlib.pyplot as plt


def logreg_train(X, Y_):
    N, D = np.shape(X)
    C = np.max(Y_) + 1 

    W = np.random.randn(C, D) 
    b = np.zeros((1, C))

    param_delta = 0.01
    param_niter = 100000

    Y = data.class_to_onehot(Y_)
    
    for i in range(param_niter):
        # Eksponencirane klasifikacijske mjere
        scores = np.dot(X, W.T) + b    # N x C
        expscores = np.exp(scores) # N x C
        
        # nazivnik softmaxa
        sumexp = np.sum(expscores, axis=1, keepdims=True)    # N x 1

        # logaritmirane vjerojatnosti razreda 
        probs = expscores / sumexp     # N x C
        logprobs = np.log(probs)  # N x C

        # gubitak
        loss  = -1/N * np.sum(Y * logprobs)     # scalar
        
        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        dL_ds = probs - Y     # N x C

        # gradijenti parametara
        grad_W = 1/N * np.dot(dL_ds.T, X)    # C x D (ili D x C)
        grad_b = 1/N * np.sum(dL_ds, axis=0, keepdims=True)    # C x 1 (ili 1 x C)

        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b

    return W, b

def logreg_classify(X, W, b):
    scores = np.dot(X, W.T) + b
    expscores = np.exp(scores)
    sumexp = np.sum(expscores)
    probs = expscores / sumexp

    return probs

def logreg_decfun(W,b):
    def classify(X):
      return np.argmax(logreg_classify(X, W, b), axis=1)
    return classify


if __name__ == "__main__":
    np.random.seed(100)
    # np.random.seed()

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 100)

    # train the model
    w, b = logreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = logreg_classify(X, w, b)
    Y = np.argmax(probs, axis=1)

    '''# report performance
    accuracy, pr, M = data.eval_perf_multi(Y, Y_)
    AP = data.eval_AP(Y_)
    print(f"accuracy: {accuracy}, precision matrix: {pr}, confusion matrix: {M}, AP: {AP}")

    '''

    decfun = logreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)
    
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()