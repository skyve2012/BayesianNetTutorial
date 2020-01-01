from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Activation, Multiply, multiply, Add, Lambda, GlobalAveragePooling2D, Reshape, Conv2D
import tensorflow.keras.initializers as keras_initializers
from tensorflow.compat.v1.keras import initializers
import tensorflow_probability as tfp
tfd = tfp.distributions

SAMPLE_PER_PASS = 100
BATCH_SIZE = 32


def gen_model():
    '''
    sample model, can change based on usage.
    '''
    
    inputs = tf.keras.layers.Input(shape=(BATCH_SIZE, 8192), dtype=tf.float32)
    x = tf.keras.layers.Flatten()(x)
    
    branch_x = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x)
    std_x = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(branch_x)
    
    x = tf.keras.layers.Dense(512,  activation=tf.nn.relu)(x)
    x = tfp.layers.DenseFlipout(128, activation=None)(x)
    x = tfp.layers.DenseFlipout(1,  activation=tf.nn.tanh)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=[x, std_x*0.05])
    
    return model

    
def ModelFunc(features, labels, mode):
    
    '''
    features: [batch size, 8192]
    labels: [batch size, label dim] # label dim = 1 for just single parameter
    mode: train, test, evaluation, controlled by the Estimator
    '''
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        train = True
    else:
        train = False

    x, y = features, labels

    
    model = gen_model()
    out_mean, out_std = model(x)
    

    final_distribution = tfd.Normal(loc=out_mean, scale=out_std + 1e-3)
    final_outputs = final_distribution.sample(SAMPLE_PER_PASS)


    ############
    #predictions
    ############
    predictions = {'predictions': tf.transpose(final_outputs, [1, 0, 2])}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    
    
    y = tf.stack([y] * SAMPLE_PER_PASS)
    
    neg_log_likelihood = tf.reduce_mean(tf.reduce_sum((final_outputs - y)**2 / (2. * (out_std + 1e-3)**2), axis=-1))
    KL = sum(model.losses)/ tf.cast(BATCH_SIZE, dtype=tf.float32)
    
    loss = tf.identity(neg_log_likelihood+KL, name='formal_loss')
    check_loss = tf.identity(tf.reduce_mean((y-final_outputs)**2), name='loss_out')
    relative_error = tf.identity(tf.reduce_mean(tf.divide(tf.abs(y-final_outputs), tf.abs(y))), name='relative_error_out')
    
    tf.summary.scalar('check_loss', check_loss)
    tf.summary.scalar('relative_error', relative_error)

    
    
    ############
    #training
    ############
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = 0.00005, beta1=0.9, beta2=0.999,epsilon=1e-08)
        train_op = optimizer.minimize(loss=loss)
        global_step = tf.compat.v1.train.get_global_step()
        update_global_step = tf.compat.v1.assign(global_step, global_step + 1, name = 'update_global_step')
        
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=tf.group(train_op, update_global_step))

    ############
    #evaluation
    ############
    
    eval_metric_ops = {
            'relative_error_test': tf.compat.v1.metrics.mean_relative_error(y, final_outputs, y),
            'mse_loss_test': tf.compat.v1.metrics.mean_squared_error(y, final_outputs)}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    
    
    
    