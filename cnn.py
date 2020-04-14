import numpy as np
import cv2
import os
import glob
import itertools
from pprint import pprint

np.random.seed(0)

class Convolution():
    def __init__(self):
        self.convolutions = 8
        self.kernels = np.random.randn(self.convolutions, 3, 3) / 9
        self.last_image = None
    
    def forward(self, image):
        self.last_image = image
        h, w = image.shape
        convolutions = np.zeros((8, h-2, w-2))
        # convolution without padding
        for k in range(self.convolutions):
            conv = np.zeros((h-2, w-2))
            for i in range(h -2):
                for j in range(w - 2):
                    region = image[i:i+3, j:j+3]
                    conv[i,j] = np.multiply(region,self.kernels[k]).sum()
            convolutions[k] = conv
        return convolutions

    def back_propagation(self, gradient):
        h, w= self.last_image.shape
        for k in range(self.convolutions):
            kernel_weight = np.zeros((3,3))
            for i in range(h -2):
                for j in range(w - 2):
                    region = self.last_image[i:i+3, j:j+3]
                    kernel_weight += gradient[k,i,j] * region
            self.kernels[k] -= 0.003 * kernel_weight
    def save_weight(self):
        np.save('convo_weights', self.kernels)

    def load_weights(self):
        self.kernels = np.load('convo_weights.npy')


class MaxPool():
    def __init__(self, size = 2):
        self.size = size
        self.kernels = 8
        self.height = 13
        self.width = 13


        self.input_shape = None
        self.last_convolutions = None

    def forward(self, convolutions):
        self.input_shape = convolutions.shape
        self.last_convolutions = convolutions
        kernels, height, width = convolutions.shape
        height //= self.size
        width //= self.size
        pool = np.zeros((kernels, height, width))
        for k in range(kernels):
            conv = convolutions[k]
            for h in range(height):
                for w in range(width):
                    pool[k,h,w] = np.argmax(conv[h * self.size : h * self.size + self.size, 
                                                 w * self.size : w * self.size + self.size])
        return pool

    def back_propagation(self, gradient):
        out = np.zeros(self.input_shape)
        convolutions = self.last_convolutions
        
        kernels, height, width = self.input_shape
        height //= self.size
        width //= self.size
        # pool = np.zeros((kernels, height, width))
        for k in range(kernels):
            conv = convolutions[k]
            for h in range(height):
                for w in range(width):
                    window = conv[h * self.size : h * self.size + self.size, 
                                  w * self.size : w * self.size + self.size]
                    max_h, max_w = np.unravel_index(window.argmax(), window.shape) 
                    out[k, max_h + self.size * h, max_w + self.size * w] = gradient[k,h,w]
        # print(convolutions[0])
        # print(out[0])
        # exit()
        return out


class Softmax():
    def __init__(self, nodes):
        self.nodes = nodes
        self.flat_len = 7688
        self.weights = np.random.randn(self.flat_len, self.nodes) / self.flat_len
        self.bias = np.zeros(nodes)

        # used for back propagation
        self.pool_shape = None
        self.last_pool = None
        self.last_f_w = 0
    def forward(self, pool):
        self.pool_shape = pool.shape
        convolutions, h, w = pool.shape
        # print(pool.shape)
        # this messy shit is to convert the pool into a 1d array where each weight to 
        # a node a "node" elemnts apart when they are ravel'd
        pool = pool.flatten().reshape(convolutions, h * w)

        # if we had 3 classes, it would produce something like this
        # e.g. [w_11, w_21, w_31, w_12, w_22, w_23]
        # this is for when we calculate the back prop, the matrix multiplication makes sense :D
        flat = pool.ravel('F')
        '''
        if (flat.reshape((convolutions, h *w), order='F').reshape(self.pool_shape) == f).all():
            print('as')
        '''
        self.last_flat_pool = flat

        assert(self.flat_len == flat.shape[0])
        self.last_f_w = np.dot(flat, self.weights) + self.bias

        # stable soft max
        exp = np.exp(self.last_f_w - np.max(self.last_f_w))
        return exp / np.sum(exp)

    def back_propagation(self, p_L_p_Predictions, label, predictions):
        '''
        used to calculate 
        ∂L / ∂w = ∂L / ∂Predictions * ∂Predictions / ∂f(w) * ∂f(w) / w
        note f(w) = weight * pool + bias
        gradient : zero vector of size number of classifications
        label : what was predicted
        predictions : vector of probabilities of size number of classifications aka softmax

        also need to return 
            ∂L / ∂input = ∂L / ∂Predictions * ∂Predictions / ∂f(w) * ∂f(w) / w
        in order to backprop the convolution layer eventaully
        '''

        label = 0
        gradient = 0
        for i, v in enumerate(p_L_p_Predictions):
            if v != 0:
                label = i
                gradient = v
                break

        # ∂predictions / ∂f(w) = ∂soft max / ∂f(w)  = ∂e^p_i / ∂sum p_j / ∂f(w)

        # ∂e^p_i / ∂sum p_j / ∂f(w)
        # source: https://deepnotes.io/softmax-crossentropy
        prob_i = predictions[label]
        p_Soft_Max = np.zeros(self.nodes)
        for j, prob_j in enumerate(predictions):
            if label == j:
                p_Soft_Max[j] = prob_i * (1 - prob_j)
            else:
                p_Soft_Max[j] = -prob_j * prob_i

        p_f_w_w = self.last_flat_pool
        p_L_p_w = gradient * np.outer(p_f_w_w, p_Soft_Max) 

        p_f_w_p_inputs = self.weights
        p_L_p_Inputs = p_f_w_p_inputs @ (gradient * p_Soft_Max)

        self.weights -= p_L_p_w * 0.003

        convolutions, h, w = self.pool_shape
        # the ungodly inverse
        return p_L_p_Inputs.reshape((convolutions, h *w), order='F').reshape(self.pool_shape)

    def save_weight(self):
        np.save('softmax_weights', self.weights)

    def load_weights(self):
        self.weights = np.load('softmax_weights.npy')

class CNN():
    def __init__(self):
        # number of outputs
        self.nodes = 6 
        self.prediction = {0: "Fist", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

        # layers
        self.layer1 = Convolution()
        self.layer2 = MaxPool()
        self.layer3 = Softmax(self.nodes)

        self.loss = 0

    def feed_forward(self, image, label = 0):
        """
        returns
        loss : calculated using cross-entropy loss
        correct : bool, if the clasified label == predicted label
        predictions  - vector of probabilities for all labels
        """
        
        # normalize image to decrease training time
        conv = self.layer1.forward(image / 255 - 0.5) # convolution
        pool = self.layer2.forward(conv) # max pool
        predictions = self.layer3.forward(pool) # soft max

        # use cross entropy loss as the loss function
        loss = -np.log(predictions[label])

        return loss, label == np.argmax(predictions), predictions
    
    def back_propagation(self, label, predictions):
        gradient = np.zeros(self.nodes)

        # want to find  ∂L / ∂weights eventaully

        # loss function is cross-entropy loss
        # ∂L / ∂Predictions
        gradient[label] = - 1 / predictions[label]

        out = self.layer3.back_propagation(gradient, label, predictions)
        out = self.layer2.back_propagation(out)
        out = self.layer1.back_propagation(out)

    def train(self, image, label):
        '''
        returns:

        loss :=  cross-entropy loss
        correct := 1 if prediction == label 0 otherwise
        '''

        loss, correct, predictions = self.feed_forward(image, label)
        self.back_propagation(label, predictions)

        return loss, correct

    def predict(self, image):
        # normalize image to decrease training time
        conv = self.layer1.forward(image / 255 - 0.5) # convolution
        pool = self.layer2.forward(conv) # max pool
        predictions = self.layer3.forward(pool) # soft max
        return self.prediction[np.argmax(predictions)], predictions

    def save_weights(self):
        print("saving weights")
        self.layer1.save_weight()
        self.layer3.save_weight()

    def load_weights(self):
        print("loading weights")
        self.layer1.load_weights()
        self.layer3.load_weights()


if __name__ == "__main__":

    labels = ["fist", "1", "2", "3", "4", "5"]
    images = []
    for idx, l in enumerate(labels):
        path = f"data/{l}/*"
        images.extend([(cv2.imread(file, cv2.IMREAD_GRAYSCALE), idx) for file in glob.glob(path)])
    
    split = int(len(images) * 0.75)
    np.random.shuffle(images)

    train_set = images[:split]
    test_set = images[split:]

    cnn = CNN()

    for epoch in range(4):
        print(f"epoch: {epoch+1}")

        np.random.shuffle(train_set)
        train_images = [im[0] for im in train_set]
        train_labels = [im[1] for im in train_set]

        # Train!
        loss = 0
        num_correct = 0
        for i, image in enumerate(zip(train_images, train_labels)):
            if i % 100 == 0:
                print(i)
            im, label = image
            l, acc = cnn.train(im, label)
            loss += l
            num_correct += acc
        print(f"Epoch done, avg loss {loss / len(train_images)} acc {num_correct/len(train_images)}")

    print('\n--- Testing the CNN ---')
    np.random.shuffle(train_set)
    test_images = [im[0] for im in test_set]
    test_labels = [im[1] for im in test_set]
    num_tests = len(test_set)
    loss = 0
    num_correct = 0
    for im, label in zip(test_images, test_labels):
        l, correct, _ = cnn.feed_forward(im, label)
        print(f"label {label} was {correct}")
        loss += l
        num_correct += correct

    print(f"Test avg loss: {loss / len(train_set)}")
    print(f"Test acc: {num_correct / len(train_set)}")

    save = input('enter y to save')
    if save == 'y':
        cnn.save_weights()
    else:
        save = input('enter y to save')
        if save == 'y':
            cnn.save_weights()
        else:
            exit()
