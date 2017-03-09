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
    train_labels = np.array(train_df['label'])
    
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
        self.model = (np.random.random([self.num_vis, self.num_hid]) *2.0 -1.0) *0.1

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

    def test_method(self):
        #test sampler
        #sample1 = self.sample(data.train_inputs[:self.mini_batch_size, :])
        #v = visualize()
        #v.show_img(sample1[23])
        #return sample1

        #test cd1
        new_model = self.cd1(self.model, data.train_inputs[:self.mini_batch_size, :])
        print "shape of model:", np.shape(new_model)
        return new_model

v = visualize()
#v.show_img(data.train_inputs[1])
s = nn(100,0.5,0.5,500,0.5,200)
s.learn_model()        
v.show_learned_features(s.model)

#print np.sum(s.test_method())
