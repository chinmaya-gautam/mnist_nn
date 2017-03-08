import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class data:
    def __init__(self):
        train_df = pd.read_csv('train.csv', sep=',', header=0)

        #normalize data and store in numpy array
        self.train_inputs = np.array(train_df.ix[:,1:].apply(lambda x:x/max(x), axis=1))
        self.train_labels = np.array(train_df['label'])
                    
class visualize:
    def __init__(self):
        pass

    def show_img(self, img_vec):
        plt.imshow(img_vec.reshape(28,28), cmap=plt.get_cmap('gray'))
        plt.show()


d = data()
v = visualize()
v.show_img(d.train_inputs[1])

        
