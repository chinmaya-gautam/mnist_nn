import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp, sqrt

np.set_printoptions(threshold='nan')

class data:
    train_df = pd.read_csv('train.csv', sep=',', header=0)
    
    #normalize data and store in numpy array
    train_inputs = np.array(train_df.ix[:,1:].apply(lambda x:x/max(x), axis=1))
    #train_inputs = np.array(train_df.ix[:,1:])
    temp = np.array(train_df['label'])
    classifier_labels = np.zeros([np.size(temp), 10])
    for i in range(np.size(temp)):
        classifier_labels[i][temp[i]] = 1
    del temp

class visualize:
    def __init__(self):
        pass

    def show_img(self, img_vec):
        plt.imshow(img_vec.reshape(28,28), cmap=plt.get_cmap('gray'))
        plt.show()

    def show_learned_features(self, features):
        num_features = np.shape(features)[1]
        num_rows = int(sqrt(num_features))
        num_cols = num_rows + num_features - num_rows*num_rows

        #print "all features shape:", np.shape(features)
        #print "num_features:", num_features
        #print "num_rows:", num_rows
        #print "num_cols:", num_cols

        fig = plt.figure()
        counter = 0
        num_features_to_plot = max(100, num_features)
        for row in range(num_rows):
            for col in range(num_cols):
                if counter >= num_features_to_plot:
                    break
                fig.add_subplot(num_rows, num_cols, counter+1)
                plt.imshow(features[:,counter].reshape(28,28), cmap=plt.get_cmap('gray'))
                plt.axis('off')
                counter += 1
        plt.show()

class nn:
    def __init__(self, num_hid,
                 lr_rbm,
                 lr_classification,
                 n_iters,
                 momentum,
                 mini_batch_size):
        self.num_vis = 784         # this is fixed as inputs are 28x28 images
        self.num_hid = num_hid     # number of hidden layers
        self.lr_rbm = lr_rbm       # learning rate of rbm
        self.lr_classification = lr_classification       #learning rate of classification
        self.n_iters = n_iters     # number of iterations
        self.momentum = momentum   # momentum
        self.mini_batch_size = mini_batch_size           #mini_batch_size
        

        self.randomness_source = np.random.random(1000*1000)
        #self.randomness_source = self.randomness_source/(max(self.randomness_source)) -0.01

        # initialize a random model
        # model = weight matrix or dimensions number of visible layers by 
        # number of hidden layers
        # x2 -1 so that it has 0 mean, 0.1 multiplier to make it small
        self.model = (np.random.random([self.num_vis, self.num_hid]) *2.0 -1.0) *0.1

        #classifier model has inputs as hiddern representation learned by rbm and output is 10 way softmax
        self.classifier_model = (np.random.random([self.num_hid, 10]) *2.0 -1.0) *0.1

    def sigmoid(self, z):
        def _sigmoid(a):
            return 1.0/(1.0 + exp(-a))
        vsigmoid = np.vectorize(_sigmoid)
        return vsigmoid(z)


    def sample(self, probabilities):
        #returns a bianry sample of the shape of probabilities matrix
        #sampled using bernoulli sampling

        #seed = int(np.random.randint(1000))
        seed = int(np.sum(probabilities))
        random_mat = np.reshape(self.randomness_source[seed:seed+np.size(probabilities)], np.shape(probabilities))        
        return np.where(probabilities > random_mat, 1, 0)

    def cd1(self, visible_data):
        #the contrastive divergence algorithm

        #shape of model: <#input_units> x <#hidden_units>
        #shape of visible_data: <#training_cases> x <#input_units>
        #shape of hidden_data: <#training_cases> x <#hidden_units>
        #shape of pos/neg grad: <#input_units> x <#hidden_units> (same as model)

        size_mini_batch = 1.0*np.shape(visible_data)[0]
        visible_data = self.sample(visible_data)

        #1. sample hidden state
        prob_hidden  = self.sigmoid(np.matmul(visible_data, self.model))
        sample_hidden = self.sample(prob_hidden)

        #get positive gradient
        #this is the outer product of visible_data and hidden_data (sampled)
        pos_grad = np.matmul(visible_data.T, sample_hidden) / size_mini_batch

        #2. reconstruction step
        prob_visible = self.sigmoid(np.matmul(sample_hidden, self.model.T))
        sample_visible = self.sample(prob_visible)

        #3. sample hidden state again
        new_hidden = self.sigmoid(np.matmul(sample_visible, self.model))
        sample_new_hidden = self.sample(new_hidden)
        #get negative gradient
        #same as before, outer product
        neg_grad = np.matmul(sample_visible.T, sample_new_hidden) / size_mini_batch

        return pos_grad - neg_grad

    def learn_model(self):
        start_of_next_mini_batch = 0
        momentum_speed = np.zeros(np.shape(self.model))
        #momentum_speed = np.random.random(np.shape(self.model))
        print "learning rate(rbm):", self.lr_rbm
        print "momentum:", self.momentum

        for i in range(self.n_iters):
            print "iteration num:", i
            mini_batch = data.train_inputs[start_of_next_mini_batch:start_of_next_mini_batch+self.mini_batch_size, :]
            start_of_next_mini_batch = (start_of_next_mini_batch + self.mini_batch_size) % np.shape(data.train_inputs)[0]
            gradient = self.cd1(mini_batch)
            #print "gradient\n-------"
            #print gradient
            momentum_speed = self.momentum * momentum_speed + gradient
            #print "momentum_spped\n***************"
            #print momentum_speed
            #print "model\n############"
            #print self.model
            self.model = self.model + self.lr_rbm * momentum_speed


    def softmax(self, data):
        #shape of data: <#training_cases> x <#classes>
        denominator = np.log(np.sum(np.exp(data),1))
        temp = data - np.reshape(denominator, [np.size(denominator), 1])
        ans = np.exp(temp)
        #print "shape of data:", np.shape(data)
        #print "shape of denominator:", np.shape(denominator)
        #print "shape of ans:", np.shape(ans)
        return ans

    def classifier_gradient(self, inputs, labels):
        #shape of inputs: <#training_cases> x <#inputs> 
        #                (inputs for classification model corresponds to hidden layer of rbm)
        #shape of labels: <#training_cases> x <#classes>
        #shape of model: <#inputs> x <#classes>
        #shape of gradient: <#inputs> x <#classes>

        #forward pass
        class_input = np.matmul(inputs, self.classifier_model)
        class_prob = self.softmax(class_input)
        
        #backward pass
        grad_at_class = -(labels - class_prob) / np.shape(inputs)[0]
        grad_at_input = - np.matmul(inputs.T, grad_at_class)

        return grad_at_input

    def learn_classification_model(self):
        start_of_next_mini_batch = 0
        momentum_speed = np.zeros(np.shape(self.classifier_model))
        #momentum_speed = np.random.random(np.shape(self.model))
        print "learning rate(classification):", self.lr_classification
        print "momentum:", self.momentum

        for i in range(self.n_iters):
            print "iteration num:", i
            mini_batch_inputs = data.classifier_inputs[start_of_next_mini_batch:start_of_next_mini_batch+self.mini_batch_size, :]
            mini_batch_labels = data.classifier_labels[start_of_next_mini_batch:start_of_next_mini_batch+self.mini_batch_size, :]
            start_of_next_mini_batch = (start_of_next_mini_batch + self.mini_batch_size) % np.shape(data.train_inputs)[0]
            gradient = self.classifier_gradient(mini_batch_inputs, mini_batch_labels)
            momentum_speed = self.momentum * momentum_speed + gradient
            self.classifier_model = self.classifier_model + self.lr_classification * momentum_speed
        
    def classify(self):
        #learn the hidden layer representation
        data.classifier_inputs = self.sigmoid(np.matmul(data.train_inputs, self.model))
        self.learn_classification_model()

    def test_method(self):
        #test sampler
        #sample1 = self.sample(data.train_inputs[:self.mini_batch_size, :])
        #v = visualize()
        #v.show_img(sample1[23])
        #return sample1

        #test cd1
        #new_model = self.cd1(self.model, data.train_inputs[:self.mini_batch_size, :])
        #print "shape of model:", np.shape(new_model)
        #return new_model

        #test classify
        self.classify()

v = visualize()
#v.show_img(data.train_inputs[1])
s = nn(100,0.5,0.5,10,0.5,200)
s.learn_model()        
s.test_method()
#v.show_learned_features(s.model)

#print np.sum(s.test_method())
