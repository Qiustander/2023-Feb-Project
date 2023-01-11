import tensorflow as tf
from .S4Layer import S4Layer
from typing import Any


class S4Block(tf.keras.Model):
    """S4 Block.
    Applies ``S4Layer()``, followed by an activation
    function, dropout, linear layer, skip connection and
    layer normalization.
    """
    def __init__(self,
                 features,
                 lmax,
                 N=64,
                 dropout=0.0,
                 bidirectional=True,
                 layer_norm=True,
                 is_sashimi=False,
                 postact=None):
        '''Initializer.
        Args:
            features (int): number of internal features
            lmax (int): length of input signal
            dropout (float): probability of elements being set to zero
            bidirectional (bool, optional): whether to use bidirectional S4layer
            layer_norm (bool, optional): whether to use layer normalization
        '''
        super().__init__()
        self.is_sashimi = is_sashimi
        self.s4_layer = S4Layer(d_model=features,
                           d_state=N,
                           l_max=lmax,
                           bidirectional=bidirectional)

        self.activation = tf.keras.layers.Activation('gelu')
        self.norm_layer = tf.keras.layers.LayerNormalization(epsilon=1e-5) \
            if layer_norm else Identity()
        self.dropout = tf.keras.layers.Dropout(dropout) \
            if dropout > 0 else Identity()
        if postact == 'glu':
            self.output_linear = [tf.keras.layers.Dense(features*2),
                                  GLU()]
        else:
            self.output_linear = tf.keras.layers.Dense(features,
                                                           activation=postact)

    def call(self, x):
        """Call the S4Block
        Args:
            x: tf.Tensor, a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``
        Returns:
            tf.Tensor, a tensor of the form ``[BATCH, SEQ_LEN, D_OUTPUT]``
        """
        xout = self.s4_layer(x)
        xout = self.dropout(self.activation(xout))
        if isinstance(self.output_linear, list):
            for layer in self.output_linear:
                xout = layer(xout)
        else:
            xout = self.output_linear(xout)
        if self.is_sashimi:
            return xout
        else:
            xout = xout + x
            xout = self.norm_layer(xout)
            return xout
        # return rearrange(self.norm_layer(rearrange(xout, 'b l k -> l b k')),
        #                  'l b k -> b l k')


class Identity(tf.keras.layers.Layer):
    """Custom implementation of a placeholder identity operator that is argument-insensitive.
        Args: (unused)
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initializer.
        """
        super(Identity, self).__init__()

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """Pass to dilated convolution 1d.
        Args:
            input: tf.Tensor, [B, T, Cin], input tensor.
        Returns:
            output: tf.Tensor, [B, T, Cin], the same as tensor.
        """

        return input


class GLU(tf.keras.layers.Layer):
    def __init__(self, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.dim = dim

    def call(self, x):
        out, gate = tf.split(x, num_or_size_splits=2, axis=self.dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x
