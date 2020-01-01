import tensorflow as tf
import numpy as np
import h5py
import random
random.seed(131)
from scipy.io import loadmat

def generator(filename, shuffle=True, batch_size = 32, min_SNR=0.1):
    
    '''
    read data in .mat format, separate them into data and labels, return dataset iterator
    
    '''
    
    # read data into numpy array
    with h5py.File(filename, 'r') as f:
        #data_x, data_y = f[u'WhitenedSignals'].value.astype(np.float32), f[u'm1m2'].value[:, :1].astype(np.float32)
        data_x, data_y = f[u'train_data'][:15000].astype(np.float32), f[u'train_label'][:15000, 2:3].astype(np.float32)



    dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))

    def _range_noise_adder_with_diff_SNRs(data_slice, label_slice):
        random_SNR = tf.random.uniform([1], dtype=data_slice.dtype, minval=min_SNR, maxval=3., name='Random_SNR')
        noisy_data_slice = data_slice / tf.reduce_max(data_slice, axis=-1, keepdims=True) \
            * random_SNR + tf.random.normal([8192], dtype=data_slice.dtype, mean=0.0, stddev=1.0)
        normalized_noisy_data_slice = noisy_data_slice / tf.sqrt(tf.nn.moments(noisy_data_slice, axes=-1, keepdims=True)[1])
        return normalized_noisy_data_slice, label_slice + tf.zeros([1])


    dataset = dataset.map(_range_noise_adder_with_diff_SNRs)

    dataset = dataset.prefetch(1000)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)

    dataset = dataset.batch(batch_size)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()
    
    return get_next
    
    
def tester_fixed_SNR(filename, shuffle=True, batch_size = 32, SNR=1.):
    
    '''
    read data in .mat format, separate them into data and labels, return dataset iterator
    
    '''
    
    
    with h5py.File(filename, 'r') as f:
        #data_x, data_y = f[u'WhitenedSignals'].value.astype(np.float32), f[u'm1m2'].value[:, :1].astype(np.float32)
        data_x, data_y = f[u'train_data'][15000:20000].astype(np.float32), f[u'train_label'][15000:20000, 2:3].astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((data_x, data_y))

    def _range_noise_adder_with_diff_SNRs(data_slice, label_slice):
        random_SNR = SNR
        noisy_data_slice = data_slice / tf.reduce_max(data_slice, axis=-1, keepdims=True) \
            * random_SNR + tf.random.normal([8192], dtype=data_slice.dtype, mean=0.0, stddev=1.0)
        normalized_noisy_data_slice = noisy_data_slice / tf.sqrt(tf.nn.moments(noisy_data_slice, axes=-1, keepdims=True)[1])
        return normalized_noisy_data_slice, label_slice + tf.zeros([1])

    dataset = dataset.map(_range_noise_adder_with_diff_SNRs)

    dataset = dataset.prefetch(1000)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=5000)

    dataset = dataset.batch(batch_size)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()
    
    return get_next


    
    
    