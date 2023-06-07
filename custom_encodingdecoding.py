import tensorflow as tf
from keras import layers

from custom_layers import *

# split the above class into two custom layers, one for the encoder and one for the decoder
class ImageEncoder(tf.keras.layers.Layer):
    def __init__(self, patch_size, internal_data_dim, conv_mult=1.0, **kwargs):
        super(ImageEncoder, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.internal_data_dim = internal_data_dim
        self.conv_mult = conv_mult
        self.external_data_dim = 3

        self.dct = SeparableDCT()
        self.flatten = tf.keras.layers.Flatten()
        self.conv1d = tf.keras.layers.Conv1D(int(self.patch_size * self.conv_mult), kernel_size=self.patch_size, strides=self.patch_size, 
                                                padding='valid', use_bias=False)
        
        self.embedding = self.add_weight(name='embedding', shape=(int((self.patch_size ** 2) * self.external_data_dim * self.conv_mult), self.internal_data_dim), 
                                        initializer=tf.keras.initializers.Orthogonal(10),
                                        # initializer=ReversibleInitializer(tf.keras.initializers.GlorotUniform(), runs=100),
                                        # constraint=ReversibleConstraint(alpha=0.01)
                                        )
        
    def call(self, inputs):
        x, zi = inputs
    
        sc = tf.shape(x)[-1]
        if sc == 1: # grayscale
            x = tf.tile(x, [1, 1, 1, 3])
        # elif sc != 3: # not grayscale or rgb # no support for rgba?
        #     raise ValueError('Unsupported number of channels: {}'.format(sc))

        # pad and patch
        x = tf.image.extract_patches(images=x, sizes=[1, self.patch_size, self.patch_size, 1], strides=[1, self.patch_size, self.patch_size, 1],
                                        rates=[1, 1, 1, 1], padding='SAME')
        patch_shapes = tf.shape(x)[1:-1] # number of patches in each dimension # we can calculate this outside based on the true data input spec
        x = tf.reshape(x, (-1, self.patch_size, int(self.patch_size * self.external_data_dim * self.conv_mult)))        
        
        x = self.dct(x)
        x = self.flatten(x)
        x = tf.expand_dims(x, axis=-1)
        x = self.conv1d(x)
        x = self.flatten(x)
        x = tf.matmul(x, self.embedding)

        outputs = x

        # patch encoding distance
        ped = create_2d_patch_encodable_dist(patch_shapes)
        zsl = tf.math.log1p(tf.cast(tf.abs(zi), tf.float32) / 16.0)
        zsd = tf.cast(tf.abs(zi), tf.float32) / 2048.0
        ped = tf.concat([ped, 
                         tf.expand_dims(tf.repeat(zsl, tf.shape(ped)[0]), axis=-1),
                         tf.expand_dims(tf.repeat(zsd, tf.shape(ped)[0]), axis=-1)
                         ], axis=-1)

        return outputs, patch_shapes, ped
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
            'internal_data_dim': self.internal_data_dim,
            'conv_mult': self.conv_mult,
            'external_data_dim': self.external_data_dim
        })
        return config
    
# image decoder
class ImageDecoder(tf.keras.layers.Layer):
    def __init__(self, patch_size, encoder_ref, conv_mult=1.0, **kwargs):
        super(ImageDecoder, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.conv_mult = conv_mult
        self.external_data_dim = 3
        self.encoder_ref = encoder_ref

        self.unflatten = layers.Reshape((-1, int(self.patch_size * self.conv_mult)))
        self.conv1d_transpose = tf.keras.layers.Conv1DTranspose(1, kernel_size=self.patch_size, strides=self.patch_size,
                                                        padding='valid', use_bias=False)
        self.idct = InverseSeparableDCT()

    def call(self, inputs):
        x, patch_shapes = inputs
        # patch shapes may be list of 2 or just a single int
        # if its an int, we can convert it to a list of 2 ints
        if isinstance(patch_shapes, int):
            patch_shapes = [patch_shapes, patch_shapes]
        elif isinstance(patch_shapes, tf.Tensor):
            patch_shapes = tf.unstack(patch_shapes)

        x = tf.matmul(x, tf.transpose(self.encoder_ref.embedding))
        x = self.unflatten(x)
        x = self.conv1d_transpose(x)
        x = tf.reshape(x, (-1, self.patch_size, self.patch_size, 3))
        x = self.idct(x)

        # cant reshape to patch_shapes in here, as we can't access ps/patch_shapes here
        x = tf.reshape(x, [-1] + patch_shapes + [self.patch_size, self.patch_size, 3])
        # reassemble image since these are non-overlapping patches, we can just transpose and reshape
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1] + [patch * self.patch_size for patch in patch_shapes] + [3])
        # will still need to handle potential padding, lets assume padding is always on the right and bottom
        # nevermind, this should only matter in direct reconstruction and similar tasks, wherin we would handle those cases separately

        outputs = x
        return outputs
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
            'conv_mult': self.conv_mult,
            'external_data_dim': self.external_data_dim
        })
        return config
    
# create a helper function to reconstruct images from patches
def reconstruct_image(patched_x, patch_shapes, patch_size):
    # reassemble image since these are non-overlapping patches, we can just transpose and reshape
    x = tf.reshape(patched_x, [-1] + patch_shapes + (patch_size, patch_size, 3))
    patches = tf.transpose(x, [0, 1, 3, 2, 4, 5]) # need to test this
    x = tf.reshape(x, (-1, patch_shapes[1] * patch_size, patch_shapes[2] * patch_size, 3))
    return x


# text encoder
class TextEncoder(tf.keras.layers.Layer):
    def __init__(self, patch_size, internal_data_dim, vocab_length, char_embedding_size, **kwargs):
        super(TextEncoder, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.internal_data_dim = internal_data_dim
        self.vocab_length = vocab_length
        self.char_embedding_size = char_embedding_size

        self.char_embedding = self.add_weight(name='char_embedding', shape=(self.vocab_length, self.char_embedding_size), 
                                        initializer=tf.keras.initializers.Orthogonal(10),
                                        )
        self.char_conv1d = tf.keras.layers.Conv1D(self.char_embedding_size, kernel_size=self.patch_size-1, strides=1,
                                                padding='same', use_bias=False)
        self.flatten = tf.keras.layers.Flatten()
        self.conv1d = tf.keras.layers.Conv1D(self.patch_size, kernel_size=self.patch_size, strides=self.patch_size, 
                                                padding='valid', use_bias=False)
        
        self.embedding = self.add_weight(name='embedding', shape=(int((self.patch_size ** 2) * self.char_embedding_size), self.internal_data_dim),
                                        initializer=tf.keras.initializers.Orthogonal(10), 
                                        )
    def call(self, inputs):
        x = inputs
        # input shape should be (batch_size, text_length, char_int)

        # pad and patch
        # arguably we might want a more intelligent form of breaking up a long sequence of text based on natural breaks,
        # then padding individual patches. But for now we'll just slice up the whole thing into patches and pad the last one
        # realistically we should use a pretrained tokenizer anyway, but this works for a proof of concept

        # pad, using patch_size * patch_size as the max length of a patch
        x = tf.pad(x, [[0, 0], [0, ((self.patch_size ** 2) - tf.shape(x)[-1]) % (self.patch_size ** 2)]], constant_values=0)
        # note, by using 0 for padding, we will need to offset the vocab by 1, so that 0 is not a valid 

        patch_shapes = [tf.shape(x)[-1] // (self.patch_size ** 2)]

        x = tf.nn.embedding_lookup(self.char_embedding, x)
        x = x + self.char_conv1d(x)

        x = self.flatten(x)
        x = tf.expand_dims(x, axis=-1)
        x = self.conv1d(x)
        x = self.flatten(x)
        x = tf.matmul(x, self.embedding)
        outputs = x
        return outputs, patch_shapes
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'vocab_length': self.vocab_length,
            'char_embedding_size': self.char_embedding_size,
            'patch_size': self.patch_size
        })
        return config

# decoder
class TextDecoder(tf.keras.layers.Layer):
    def __init__(self, char_embedding_size, patch_size, encoder_ref, **kwargs):
        super(TextDecoder, self).__init__(**kwargs)
        self.char_embedding_size = char_embedding_size
        self.patch_size = patch_size
        self.encoder_ref = encoder_ref

        self.unflatten = layers.Reshape((-1, self.patch_size))
        self.conv1d_transpose = tf.keras.layers.Conv1DTranspose(1, kernel_size=self.patch_size, strides=self.patch_size,
                                                        padding='valid', use_bias=False)
        
        self.rechar_conv1d = tf.keras.layers.Conv1DTranspose(self.char_embedding_size, kernel_size=self.patch_size-1, strides=1,
                                                padding='same', use_bias=False)
        
    def call(self, inputs):
        x, patch_shapes = inputs
        # if patch shapes is a list, just take the first value, and divide by patch_size
        if isinstance(patch_shapes, list):
            patch_shapes = patch_shapes[0] // self.patch_size
        else:
            patch_shapes = patch_shapes // self.patch_size

        x = tf.matmul(x, tf.transpose(self.encoder_ref.embedding))

        x = self.unflatten(x)
        x = self.conv1d_transpose(x)

        x = tf.reshape(x, (-1, self.patch_size * self.patch_size, self.char_embedding_size))
        x = x - self.rechar_conv1d(x)
        x = tf.reshape(x, (-1, self.patch_size, self.patch_size, self.char_embedding_size))
        x = tf.matmul(x, tf.transpose(self.encoder_ref.char_embedding))
        
        x = tf.nn.softmax(x, axis=-1)

        # reassemble patches
        x = tf.reshape(x, [-1, patch_shapes * (self.patch_size ** 2), self.encoder_ref.vocab_length])


        outputs = x
        return outputs
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'char_embedding_size': self.char_embedding_size,
            'patch_size': self.patch_size,
            'encoder_ref': self.encoder_ref
        })
        return config

class Imagenet1kClassificationEncoder(tf.keras.layers.Layer):
    def __init__(self, internal_data_dim, **kwargs):
        super(Imagenet1kClassificationEncoder, self).__init__(**kwargs)
        self.internal_data_dim = internal_data_dim

        self.embedding = self.add_weight(name='embedding', shape=(1000, self.internal_data_dim),
                                         # reversible
                                            initializer=tf.keras.initializers.Orthogonal(10),
                                            trainable=True)
        
    def call(self, inputs):
        x, zi = inputs
        ex = tf.matmul(tf.expand_dims(x, axis=0), self.embedding)

        ped = tf.zeros((1, 4))
        zi = tf.reshape(tf.abs(zi), (1, 1))
        zsl = tf.math.log1p(tf.cast(tf.abs(zi), tf.float32) / 16.0)
        zsd = tf.cast(tf.abs(zi), tf.float32) / 2048.0
        ped = tf.concat([ped, zsl, zsd], axis=-1)

        return ex, [1,], ped
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'internal_data_dim': self.internal_data_dim
        })
        return config
        
class Imagenet1kClassificationDecoder(tf.keras.layers.Layer):
    def __init__(self, encoder_ref, **kwargs):
        super(Imagenet1kClassificationDecoder, self).__init__(**kwargs)
        self.encoder_ref = encoder_ref
        
    def call(self, inputs):
        x, patch_shapes = inputs
        x = tf.matmul(x, self.encoder_ref.embedding, transpose_b=True)
        x = tf.nn.softmax(x, axis=-1)
        return x
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'encoder_ref': self.encoder_ref
        })
        return config

# function that takes in string names of datatypes, and outputs a list of encoders and a list of decoders for those datatypes
def get_encoders_and_decoders(data_size, modes, **kwargs):
    encoders = []
    decoders = []
    ordered_modes = []
    for data_type in modes:
        if data_type == 'image':
            encoder = ImageEncoder(patch_size=kwargs.get('patch_size'), 
                                  internal_data_dim=data_size, 
                                  conv_mult=kwargs.get('conv_mult'),
                                  )
            decoder = ImageDecoder(patch_size=kwargs.get('patch_size'),
                                    encoder_ref=encoder,
                                    conv_mult=kwargs.get('conv_mult')
                                    )
        elif data_type == 'text':
            encoder = TextEncoder(patch_size=kwargs.get('patch_size'), 
                                  internal_data_dim=data_size, 
                                  vocab_length=kwargs.get('vocab_length'), 
                                  char_embedding_size=kwargs.get('char_embedding_size')
                                  )
            decoder = TextDecoder(char_embedding_size=kwargs.get('char_embedding_size'),
                                    patch_size=kwargs.get('patch_size'),
                                    encoder_ref=encoder
                                    )
        elif data_type == 'imagenet1k_classification':
            encoder = Imagenet1kClassificationEncoder(internal_data_dim=data_size)
            decoder = Imagenet1kClassificationDecoder(encoder_ref=encoder)
        else:
            raise ValueError('data_type not recognized')
        encoders.append(encoder)
        decoders.append(decoder)
        ordered_modes.append(data_type)
        
    # take the lists and make them dictionaries with the data types as keys
    encoders = dict(zip(ordered_modes, encoders))
    decoders = dict(zip(ordered_modes, decoders))
    return encoders, decoders      


def create_2d_patch_encodable_dist(patch_dims): # could rewrite faster to be 2d specific
    output_list = []
    for i in range(len(patch_dims)):
        di = patch_dims[i]
        di_f = tf.cast(di, tf.float32) - 1.0
        di_grid = tf.cast(tf.range(di), tf.float32)
        di_centered = di_grid - (di_f / 2.0)
        di_log_scaled = tf.sign(di_centered) * (tf.math.log1p(tf.abs(di_centered / 16.0)) / 2.0)
        di_direct_scaled = di_centered / (di_f / 2.0)
        n = 0
        for j in range(len(patch_dims)):
            dj = patch_dims[j]
            if j < i:
                di_log_scaled = tf.repeat(tf.expand_dims(di_log_scaled, axis=n), dj, axis=n)
                di_direct_scaled = tf.repeat(tf.expand_dims(di_direct_scaled, axis=n), dj, axis=n)
                n += 1
            elif j > i:
                di_log_scaled = tf.repeat(tf.expand_dims(di_log_scaled, axis=-1), dj, axis=-1)
                di_direct_scaled = tf.repeat(tf.expand_dims(di_direct_scaled, axis=-1), dj, axis=-1)
        
        output_list.append(di_log_scaled)
        output_list.append(di_direct_scaled)

    output = tf.reshape(tf.stack(output_list, axis=-1), (-1, len(patch_dims) * 2))
    return output
