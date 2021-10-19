import numpy as np

class CustomHeteroLinkDataLoader():

    def __init__(self, data, batch_size, shuffle = False):
        self.data = data
        self.batch_size = batch_size
        self.index = 0
        self.shuffle = shuffle
        self.max_len = len(self.data[('user','rated','movie')]['edge_label'])
        indices = np.arange(self.max_len)
        self.data_edge_labels = self.data[('user','rated','movie')]['edge_label'][indices]
        self.data_edge_labels_index = self.data[('user','rated','movie')]['edge_label_index'][:,indices]
        
    def reshuffle(self):
        indices = np.random.permutation(self.max_len)
        self.data_edge_labels = self.data[('user','rated','movie')]['edge_label'][indices]
        self.data_edge_labels_index = self.data[('user','rated','movie')]['edge_label_index'][:,indices]


    def __iter__(self):
        if self.shuffle:
            self.reshuffle()
        
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.max_len:
           edge_labels = self.data_edge_labels[self.index:self.index + self.batch_size]
           edge_labels_index = self.data_edge_labels_index[:,self.index:self.index + self.batch_size]
           self.index += self.batch_size
           
           return self.data, edge_labels, edge_labels_index
        else:
            raise StopIteration


    def __len__(self):
        return int(self.max_len/self.batch_size)