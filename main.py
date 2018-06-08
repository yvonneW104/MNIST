import pandas as pd
import numpy as np
import network
import optims
from solver import *
import gzip

import matplotlib.pyplot as plt

def data_load (percentage = 0.8, dim=4) :
    file_names = [['X_train', 'train-images-idx3-ubyte.gz'],
				 ['y_train', 'train-labels-idx1-ubyte.gz'],
				 ['X_test', 't10k-images-idx3-ubyte.gz'],
				 ['y_test', 't10k-labels-idx1-ubyte.gz']]

    mnist_data = {}
    for pair in file_names :
        with gzip.open(pair[1], 'rb') as f:
            if pair[0].startswith('X') :
                if dim == 4:
                    mnist_data[pair[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
                else:
                    mnist_data[pair[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
            else :	# label
                mnist_data[pair[0]] = np.frombuffer(f.read(), np.uint8, offset=8)


    idx = np.arange(mnist_data['X_train'].shape[0])
    np.random.shuffle(idx)
    length = idx.shape[0]


    train_idx = idx[:int(length*percentage)]
    val_idx = idx[int(length*percentage):]


    mnist_data['X_val'] = mnist_data['X_train'][val_idx]
    mnist_data['y_val'] = mnist_data['y_train'][val_idx]
    mnist_data['X_train'] = mnist_data['X_train'][train_idx]
    mnist_data['y_train'] = mnist_data['y_train'][train_idx]

    # print(mnist_data['y_train'])
    return mnist_data



def main():
    data = data_load()
    # for LR
    # data = data_load(dim=2)
    print('data_over')
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    classes = 10

    def one_hot(y, class_now):
        y_ = np.zeros(y.shape)
        y_[y == class_now] = 1
        y_[y != class_now] = -1
        return y_


    SVM kernel
    Since SVM do not use standard SGD or SGD_momentum or Adam, this model is excluded from the solver
    class_test = [0,1,2,3,4,5,6,7,8,9]

    index_in = np.arange(X_train.shape[0])
    np.random.shuffle(index_in)
    index_ = index_in[:2000]
    scores = np.zeros((y_test.shape[0],classes))
    acc_list = []

    # data prepare
    X_train = X_train/255.0
    X_test = X_test/255.0

    ites = [50,200, 500, 1000]
    for ite in ites:
        for class_ in class_test:
            # model SVM_Kernel has two types, linear kernel and RBF
            model = network.SVM_Kernel(max_ite= ite,kernel_name = 'linear', C = 0.3, batch_size = 20, gamma_in = 0.05)
            theta,b = model.train(X_train[index_],one_hot(y_train[index_],class_))
            print("class {} done".format(class_))
            scores[:,class_] = model.test(X_train[index_],one_hot(y_train[index_],class_),X_test,one_hot(y_test,class_),theta,b)

        label = np.argmax(scores,axis=1)
        print(label[:40])
        print(y_test[:40])
        acc = np.sum(label == y_test) / y_test.shape[0]
        print("acc = ", acc)

        acc_list.append(acc)

    print(acc_list)
    plt.plot(ites,acc_list)
    plt.show()


    # All the other model can be put in solver
    lrs = [5e-8]

    update_rules = ['sgd', 'sgd_momentum','adam']

    for update_rule in update_rules:
        # model = network.LogisticRegression(input_dim=X_train.shape[1], reg= 0.5, reg_type= 'l2')
        # model = network.FullyConnectedNet(hidden_dims=[256, 128], weight_scale=0.01)
        model = network.ConvNet(weight_scale=1e-2)
        # model = network.SVM(input_dim=X_train.shape[1])
        solver = Solver(model, data, num_epochs=10, batch_size=200, update_rule=update_rule,
                        optim_config={'learning_rate':1e-3},
                        verbose=True,
                        print_every= 50
                        )
        solver.train()

        y_prob = model.loss(X_test)
        y_prob = np.argmax(y_prob, axis=1)

        test_acc = np.mean(y_prob == y_test)
        print('test_acc = ', test_acc)

        plt.title('Training loss')
        plt.xlabel('Iteration')
        plt.plot(solver.loss_history)
        # plt.hold(True)

        # plt.figure()
        # plt.title('Train / Validation accuracy')
        # plt.xlabel('Epoch')
        # plt.plot(solver.train_acc_history, 'o-', label='train acc')
        # plt.plot(solver.val_acc_history, 'o-', label='val acc')
        # plt.legend(loc='lower right', ncol=4)
    plt.legend(['sgd','sgd_momentum','adam'])
    plt.show()



if __name__ == '__main__':
    main()



