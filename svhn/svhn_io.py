import numpy as np
import scipy.io as sio
import h5py

def load_svhn():
    data = h5py.File('svhn_32x32.hdf5', 'r')

    x_train = data['x_train'][...]
    y_train = data['y_train'][...]

    x_test = data['x_test'][...]
    y_test = data['y_test'][...]

    #Normalization
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return (x_train, y_train), (x_test, y_test)
    
def load_svhn_mat(filename):
    data = sio.loadmat(filename)

    x = data["X"]
    x = np.transpose(x, (3, 0, 1, 2))

    y = data["y"]
    y[y == 10] = 0

    print("Loaded SVHN MAT data - x {} y {}".format(x.shape, y.shape))
    print("X: {} {}".format(x.min(), x.max()))

    return x, y

# Preprocess and save data as HDF5
if __name__ == "__main__":
    x_train, y_train = load_svhn_mat('train_32x32.mat')
    x_test, y_test = load_svhn_mat('test_32x32.mat')

    output = h5py.File('svhn_32x32.hdf5')
    output.create_dataset('x_train', data=x_train, dtype=np.uint8, compression='gzip', compression_opts=9, chunks=(100, 32, 32, 3))
    output.create_dataset('y_train', data=y_train, dtype=np.uint8, compression='gzip', compression_opts=9, chunks=(100, 1))

    output.create_dataset('x_test', data=x_test, dtype=np.uint8, compression='gzip', compression_opts=9, chunks=(100, 32, 32, 3))
    output.create_dataset('y_test', data=y_test, dtype=np.uint8, compression='gzip', compression_opts=9, chunks=(100, 1))

    output.close()