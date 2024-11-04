import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W

def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return  1 / (1 + np.exp(-z))

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    #-------------------------- feature selection ----------------------------------------------------#
    selected_features = [i for i in range(train_data.shape[1]) if len(np.unique(train_data[:,i])) > 1]
    train_data = train_data[:, selected_features]
    validation_data = validation_data[:, selected_features]
    test_data = test_data[:, selected_features]

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features

def nnFeedForward(w1, w2, feature_vector, return_hidden=False):
    # Add bias activation to the feature vector as 1 ---> (n_input + 1, 1)
    feature_vector_with_bias = np.append(feature_vector, 1)

    # Compute activations of the hidden layer ---> (n_hidden, 1)
    hidden_activations = sigmoid(np.matmul(w1, feature_vector_with_bias))

    # Add bias activation to hidden layer as 1 ---> (n_hidden + 1, 1)
    hidden_activations_with_bias = np.append(hidden_activations, 1)

    if return_hidden:
        return hidden_activations_with_bias
    
    # Compute activations of the output layer ---> (n_class, 1)
    output_activations = sigmoid(np.matmul(w2, hidden_activations_with_bias))

    return output_activations

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])

    for feature_vector in data:
        O = nnFeedForward(w1, w2, feature_vector, return_hidden=False)
        labels = np.append(labels, np.argmax(O))

    return labels

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    obj_val = 0
    obj_grad = np.array([])

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    def encode_label(label):
        # Encode label according to 1 of K scheme
        return np.array([1 if i == label else 0 for i in range(n_class)])

    def log_loss(encoded_class_label, prediction):
        # Compute log-loss for a single class assuming label has already been 1 of K coded
        return -((encoded_class_label * np.log(prediction)) + ((1-encoded_class_label) * np.log(1-prediction)))

    def compute_image_log_loss(w1, w2, feature_vector, encoded_label):
      # Feed forward image feature vector to obtain model prediction
      prediction_vector = nnFeedForward(w1, w2, feature_vector)

      # Sum losses for all n_class possible labels, assuming label has been 1 of K encoded
      return np.sum([log_loss(class_label, pred) for class_label, pred in zip(encoded_label, prediction_vector)])
    
    def compute_regularization_error(w1, w2, n, lambdaval):
        return (lambdaval / (2 * n)) * (np.sum(np.square(w1.flatten())) + np.sum(np.square(w2.flatten())))

    def compute_single_image_grad_w2(hidden_activations_with_bias, output_activations, encoded_label):
        # Define delta as difference between predicted and true label probabilities ---> (n_class,)
        delta = output_activations - encoded_label

        # Reshape 1D arrays into 2D arrays to allow for matrix multiplications
        delta_2D = delta.reshape(delta.shape[0], 1)
        hidden_activations_2D = hidden_activations_with_bias.reshape(hidden_activations_with_bias.shape[0], 1)

        # Compute gradient as matrix multiplication of delta and the hidden activations ---> (n_class, n_hidden + 1)
        single_grad_w2 = np.matmul(delta_2D, hidden_activations_2D.T)

        return single_grad_w2

    def compute_single_image_grad_w1(w2, feature_vector, hidden_activations_with_bias, output_activations, encoded_label):
        # Add bias activation to the feature vector as 1 ---> (n_input + 1,)
        feature_vector_with_bias = np.append(feature_vector, 1)

        # Define delta as difference between predicted and true label probabilities ---> (n_class,)
        delta = output_activations - encoded_label

        # Reshape 1D arrays into 2D arrays to allow for matrix multiplications
        delta_2D = delta.reshape(delta.shape[0], 1)
        feature_vector_2D = feature_vector_with_bias.reshape(feature_vector_with_bias.shape[0], 1)
        hidden_activations_2D = hidden_activations_with_bias.reshape(hidden_activations_with_bias.shape[0], 1)

        # Compute linear combinations term (sum delta_l * w_lj) matrix ---> (n_hidden + 1, 1)
        linear_combinations = np.matmul(w2.T, delta_2D)

        # Compute product terms matrix ((1-Z) * Z * linear combinations), leaving out hidden layer bias weight ---> (n_hidden, 1)
        product = np.multiply(np.ones(hidden_activations_2D[:-1].shape) - hidden_activations_2D[:-1], hidden_activations_2D[:-1])
        product = np.multiply(product, linear_combinations[:-1])

        # Multiply product by feature vector matrices to compute gradient ---> (n_hidden, n_input + 1)
        single_grad_w1 = np.matmul(product, feature_vector_2D.T)

        return single_grad_w1
    
    # Encode training true labels in 1 of K scheme
    encoded_training_label = np.array([encode_label(label) for label in training_label])

    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)

    if lambdaval > 0:
        # If regularizing, initialize obj_val and gradient matrices to include regularization contributions
        obj_val = compute_regularization_error(w1, w2, training_data.shape[0], lambdaval)
        grad_w1 = lambdaval * w1
        grad_w2 = lambdaval * w2

    # Iterate over each training example
    for feature_vector, encoded_label in zip(training_data, encoded_training_label):

        # Add log_loss contribution to total error for this training example
        obj_val += compute_image_log_loss(w1, w2, feature_vector, encoded_label)

        # Compute activations for each layer for this training example
        hidden_activations_with_bias = nnFeedForward(w1, w2, feature_vector, return_hidden=True)
        output_activations = nnFeedForward(w1, w2, feature_vector, return_hidden=False)

        # Add contributions to gradient matrices for this training example
        grad_w1 = grad_w1 + compute_single_image_grad_w1(w2, feature_vector, hidden_activations_with_bias, output_activations, encoded_label)
        grad_w2 = grad_w2 + compute_single_image_grad_w2(hidden_activations_with_bias, output_activations, encoded_label)
    
    # Divide error and gradients by number of training examples to get the mean
    n = training_data.shape[0]
    obj_val /= n
    grad_w1 /= n
    grad_w2 /= n

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)

    print("Total loss = {val:.4f}......".format(val=obj_val))

    return (obj_val, obj_grad)