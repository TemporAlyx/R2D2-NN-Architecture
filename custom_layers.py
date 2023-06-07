import tensorflow as tf
from keras import layers
from keras import backend as K


class SeparableDCT(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SeparableDCT, self).__init__(**kwargs)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        x = inputs
        if len(input_shape) > 2:
            # Apply DCT to each convolutional dimension in the input data
            for i in range(1, len(input_shape) - 1):
                # reshape the dimension to be the last one # this is one disgusting transpose
                perm = tf.concat([tf.range(i), [len(input_shape) - 1], tf.range(i+1, len(input_shape) - 1), [i]], axis=0)
                x = tf.transpose(x, perm=perm)
                # force float32 for the tf.signal.dct function
                # x = tf.cast(x, tf.float32)
                x = tf.signal.dct(x, type=2, norm='ortho')
                # x = tf.cast(x, inputs.dtype)
            # return the data dim to its original position
            perm = tf.concat([[0, len(input_shape) - 1], tf.range(2, len(input_shape) - 1), [1]], axis=0)
            x = tf.transpose(x, perm=perm)
        return x
    
class InverseSeparableDCT(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InverseSeparableDCT, self).__init__(**kwargs)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        x = inputs
        if len(input_shape) > 2:
            # Apply iDCT to each convolutional dimension in the input data
            for i in range(1, len(input_shape) - 1):
                # reshape the dimension to be the last one # this is one disgusting transpose
                perm = tf.concat([tf.range(i), [len(input_shape) - 1], tf.range(i+1, len(input_shape) - 1), [i]], axis=0)
                x = tf.transpose(x, perm=perm)
                # force float32 for the tf.signal.idct function
                # x = tf.cast(x, tf.float32)
                x = tf.signal.idct(x, type=2, norm='ortho')
                # x = tf.cast(x, inputs.dtype)
            # return the data dim to its original position
            perm = tf.concat([[0, len(input_shape) - 1], tf.range(2, len(input_shape) - 1), [1]], axis=0)
            x = tf.transpose(x, perm=perm)
        return x
    

class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, max_lr, warmup_steps, alpha):
        super().__init__()
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha

        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(max_lr, total_steps - warmup_steps)

    def __call__(self, step):
        epoch_step = step % self.total_steps
        linear_warmup = self.max_lr * epoch_step / self.warmup_steps
        cosine_decay = self.cosine_decay(epoch_step - self.warmup_steps)
        return tf.where(step < self.warmup_steps, linear_warmup, cosine_decay)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha
        })
    
# APTx activation function, (1 + tanh(x)) * 0.5x
@tf.function
def aptx(x):
    return (1.0 + tf.math.tanh(x)) * (0.5 * x)
