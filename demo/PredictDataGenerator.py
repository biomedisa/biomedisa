import numpy as np
import keras

class PredictDataGenerator(keras.utils.Sequence):
    def __init__(self, img, position, list_IDs, batch_size=32, dim=(32,32,32),
                 dim_img=(32,32,32), n_channels=1):
        'Initialization'
        self.dim = dim
        self.dim_img = dim_img
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.img = img
        self.position = position
        self.n_channels = n_channels
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # get patch indices
            k = ID // (self.dim_img[1]*self.dim_img[2])
            rest = ID % (self.dim_img[1]*self.dim_img[2])
            l = rest // self.dim_img[2]
            m = rest % self.dim_img[2]

            # get patch
            X[i,:,:,:,0] = self.img[k:k+self.dim[0],l:l+self.dim[1],m:m+self.dim[2]]
            if self.n_channels == 2:
                X[i,:,:,:,1] = self.position[k:k+self.dim[0],l:l+self.dim[1],m:m+self.dim[2]]

        return X
