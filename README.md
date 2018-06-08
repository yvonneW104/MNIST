# ECE210 Project: Digit classification on MNIST dataset


Tianxue Chen, Lu Ren, Ying Wang

## Dataset

## Scripts
This projects consists of the following scripts:

- main.py – main function, de

- solver.py - encapsulates all the logic necessary for training classification models. By constructing a Solver instance, passing the model, dataset, and various optoins, the modle could be trained. And it will check the training and validation accuracy per epoch, and save the best parameter with highes accuracy. In addition, the instance variable solver.loss_history will contain a list of all losses encountered during training and the instance variables solver.train_acc_history and solver.val_acc_history will be lists of the accuracies of the model on the training and validation set at each epoch
  - solver.train: run optimization to train the model
  - solver._step: make a single gradient update. This is called by train() and should not be called manually
  - solver.check_accuracy: check accuracy of the model on the provided data

- network.py - creates several networks, including LR, SVM, SVM with kernel, FCNet and CNN. To be specific, SVM with kernel do not use regular gradient descent method, so the training part for SVM_Kernel do in this class

- layers.py - defines forward and backward computation for fully connected layer, conv layer, max pooling layer, activation layer, and etc

- layer_utils.py - defines few convenience layer, such as conv_relu_forward which performs a convolution followed by a ReLU

- optims.py - defines three optimization methods: SGD, SGD Momentum, Adam

- cs231n - This folder includes the fast version implementation of convolution layer. Thanks to Standford CS231n course, we speed up at least 500x for backward pass than our naive convolution layer. The fast convolution implementation depends on a Cython extension; to compile it need to run the following from the cs231n directory:
  ```
  python setup.py build_ext --inplace
  ```
