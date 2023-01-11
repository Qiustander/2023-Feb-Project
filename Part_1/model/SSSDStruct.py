import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from .S4Model import S4Block, Identity, GLU
from einops import rearrange


############################ Block for SSSDS4
class SSSDS4Block(tf.keras.Model):
    """WaveNet Block.
    """
    def __init__(self, res_channels, skip_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSDS4Block, self).__init__()
        self.res_channels = res_channels
        self.s4_dropout = s4_dropout
        self.s4_bidirectional = s4_bidirectional
        self.s4_layernorm = s4_layernorm
        self.s4_d_state = s4_d_state

        self.fc_t = tf.keras.layers.Dense(self.res_channels)

        self.firstS4 = S4Block(features=2 * self.res_channels,
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)

        self.conv_layer = WeightNormalization(tf.keras.layers.Conv1D(2 * self.res_channels,
                                kernel_size=3, padding='same',
                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                              bias_initializer=tf.keras.initializers.HeNormal()))

        self.secondS4 = S4Block(features=2 * self.res_channels,
                           lmax=s4_lmax,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm)

        self.cond_conv = WeightNormalization(tf.keras.layers.Conv1D(2 * self.res_channels,
                                kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(),
                                             bias_initializer=tf.keras.initializers.HeNormal()))

        self.res_conv = WeightNormalization(tf.keras.layers.Conv1D(self.res_channels,
                                kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(),
                                            bias_initializer=tf.keras.initializers.HeNormal()))

        self.skip_conv = WeightNormalization(tf.keras.layers.Conv1D(skip_channels,
                                kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(),
                                             bias_initializer=tf.keras.initializers.HeNormal()))

    def call(self, input_data):
        """Pass Block.
        Args:
            input_data, Tuple,
                x: tf.Tensor, [B, T, C(=channels)], input tensor.
                cond: tf.Tensor, [B, T, 2C], embedding tensor for noise schedules.
                diffusion_step_embed: tf.Tensor, [B, T], conditions.
        Returns:
            residual: tf.Tensor, [B, T, C], output tensor for residual connection.
            skip: tf.Tensor, [B, T, C], output tensor for skip connection.
        """
        signal, cond, diffusion_step_embed = input_data
        B, L, C = signal.shape # [B, T, K]
        assert C == self.res_channels

        part_t = self.fc_t(diffusion_step_embed)
        h = signal + part_t[:, None]

        h = self.conv_layer(h)
        h = self.firstS4(h)

        # print('first S4 layer time: %.2f' %(end - start))
        assert cond is not None
        cond = self.cond_conv(cond)
        h += cond
        h = self.secondS4(h)
        # print('second S4 layer time: %.2f' % (end - start))
        out = tf.math.tanh(h[..., :self.res_channels]) * \
              tf.math.sigmoid(h[..., self.res_channels:])
        res = self.res_conv(out)
        # assert x.shape == res.shape
        skip = self.skip_conv(out)

        output = (signal + res) * tf.math.sqrt(0.5)

        return output, skip  # normalize for training stability


############################ Structure for SSSDS4
class SSSDStuct(tf.keras.Model):
    """WaveNet structure.
    """
    def __init__(self, config):
        super(SSSDStuct, self).__init__()
        self.num_res_layers = config['num_res_layers']
        res_channels = config['res_channels']
        skip_channels = config['skip_channels']
        self.diffusion_step_embed_dim_in = config['diffusion_step_embed_dim_in']
        diffusion_step_embed_dim_mid = config['diffusion_step_embed_dim_mid']
        diffusion_step_embed_dim_out = config['diffusion_step_embed_dim_out']
        s4_lmax = config['s4_len']
        s4_d_state = config['s4_d_state']
        s4_dropout = config['s4_dropout']
        s4_bidirectional = config['s4_bidirectional']
        s4_layernorm = config['s4_layernorm']

        # for diffusion time embedding
        self.fc_t1 = tf.keras.layers.Dense(diffusion_step_embed_dim_mid)
        self.fc_t2 = tf.keras.layers.Dense(diffusion_step_embed_dim_out)

        self.residual_blocks = [SSSDS4Block(res_channels, skip_channels,
                                                   s4_lmax=s4_lmax,
                                                   s4_d_state=s4_d_state,
                                                   s4_dropout=s4_dropout,
                                                   s4_bidirectional=s4_bidirectional,
                                                   s4_layernorm=s4_layernorm) for i in range(self.num_res_layers) ]

    def call(self, input_data):
        """Forwad of SSSD Structure
        Args:
            input_data: tuple,
                x: tf.Tensor, [B, T, res_channels], multivariate time series feature map
                conditional: tf.Tensor, [B, T, K], original multivariate time series
                diffusion_steps: tf.Tensor, [B, 1], diffusion time step
        Returns:
                y: tf.Tensor, [B, T, K], predicted output.
        """
        x, conditional, diffusion_steps = input_data

        diffusion_step_embed = self.calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = tf.keras.activations.swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = tf.keras.activations.swish(self.fc_t2(diffusion_step_embed))

        skip = []
        for layer in self.residual_blocks:
            x, skip_n = layer((x, conditional, diffusion_step_embed))
            skip.append(skip_n)

        skip = tf.reduce_sum(skip, axis=0) * tf.math.sqrt(1.0 / self.num_res_layers)

        return skip

    def calc_diffusion_step_embedding(self, diffusion_steps, diffusion_step_embed_dim_in):
        """Embed a diffusion step $t$ into a higher dimensional space
        E.g. the embedding vector in the 128-dimensional space is
        [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]
        Args:
            diffusion_steps: tf.Tensor, diffusion steps for batch data
            diffusion_step_embed_dim_in: int, dimensionality of the
                                    embedding space for discrete diffusion steps
        Returns:
            tf.Tensor, [batch, diffusion_step_embed_dim_in], embedding vectors.
        """
        assert not diffusion_step_embed_dim_in % 2
        half_dim = diffusion_step_embed_dim_in // 2
        exp = tf.math.exp(- tf.range(0., half_dim) * tf.math.log(10000.) / (half_dim - 1))
        # _embed = exp[None] * tf.cast(diffusion_steps[:, None], tf.float32)
        _embed = diffusion_steps * exp
        return tf.concat([tf.sin(_embed), tf.cos(_embed)], axis=-1)


############################ Structure for SSSDSA
class SSSDSA(tf.keras.Model):
    """WaveNet structure.
    """
    def __init__(self, config):
        super(SSSDSA, self).__init__()
        self.num_layers = config['num_layers']
        res_channels = config['res_channels']
        self.diffusion_step_embed_dim_in = config['diffusion_step_embed_dim_in']
        diffusion_step_embed_dim_mid = config['diffusion_step_embed_dim_mid']
        diffusion_step_embed_dim_out = config['diffusion_step_embed_dim_out']
        s4_lmax = config['s4_len']
        s4_d_state = config['s4_d_state']
        s4_dropout = config['s4_dropout']
        s4_bidirectional = config['s4_bidirectional']
        s4_layernorm = config['s4_layernorm']
        self.use_unet = config['use_unet']
        in_channels = config['in_channels']
        self.dropout = 0.
        ff = 2
        pool = [2, 2]
        expand = 2

        # for diffusion time embedding
        self.fc_t1 = tf.keras.layers.Dense(diffusion_step_embed_dim_mid)
        self.fc_t2 = tf.keras.layers.Dense(diffusion_step_embed_dim_out)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    #   #   #   #   #   #   #   #   #   Define Processing Block
        def s4_block(dim, stride):
            layer = S4Block(
                features=dim,
                lmax=s4_lmax//stride,
                N=s4_d_state,
                bidirectional=s4_bidirectional,
                dropout=s4_dropout,
                layer_norm=s4_layernorm,
                is_sashimi=True,
                postact='glu')
            return SSSDSABlock(res_channels=dim,
                layer=layer,
                dropout=self.dropout,
                in_channels=in_channels,
                stride=stride
            )

        def ff_block(dim, stride):
            layer = FFBlock(
                d_model=dim,
                expand=ff,
                dropout=self.dropout,
            )
            return SSSDSABlock(res_channels=dim,
                layer=layer,
                dropout=self.dropout,
                in_channels=in_channels,
                stride=stride
            )

#   #   #   #   #   #   #   #   #   Define UNet structure
        # Down blocks
        d_layers = []
        for i, p in enumerate(pool):
            if self.use_unet:
                # Add blocks in the down layers
                for _ in range(self.num_layers):
                    if i == 0:
                        d_layers.append(s4_block(res_channels, 1))
                        if ff > 0: d_layers.append(ff_block(res_channels, 1))
                    elif i == 1:
                        d_layers.append(s4_block(res_channels, p))
                        if ff > 0: d_layers.append(ff_block(res_channels, p))
            # Add sequence downsampling and feature expanding
            d_layers.append(DownPool(res_channels, expand, p))
            res_channels *= expand

        # Center block
        c_layers = []
        for _ in range(self.num_layers):
            c_layers.append(s4_block(res_channels, pool[1] * 2))
            if ff > 0: c_layers.append(ff_block(res_channels, pool[1] * 2))

        # Up blocks
        u_layers = []
        for i, p in enumerate(pool[::-1]):
            block = []
            res_channels //= expand
            block.append(UpPool(res_channels * expand, expand, p, causal=False))

            for _ in range(self.num_layers):
                if i == 0:
                    block.append(s4_block(res_channels, pool[0]))
                    if ff > 0: block.append(ff_block(res_channels, pool[0]))

                elif i == 1:
                    block.append(s4_block(res_channels, 1))
                    if ff > 0: block.append(ff_block(res_channels, 1))

            u_layers.append(block)

        self.down_layers = d_layers
        self.up_layers = u_layers
        self.cent_layers = c_layers

    def call(self, input_data):
        """Forwad of SSSD Structure
        Args:
            input_data: tuple,
                x: tf.Tensor, [B, T, res_channels], multivariate time series feature map
                conditional: tf.Tensor, [B, T, K], original multivariate time series
                diffusion_steps: tf.Tensor, [B, 1], diffusion time step
        Returns:
                y: tf.Tensor, [B, T, K], predicted output.
        """
        x, conditional, diffusion_steps = input_data

        diffusion_step_embed = self.calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = tf.keras.activations.swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = tf.keras.activations.swish(self.fc_t2(diffusion_step_embed))

        # Down blocks
        outputs = []
        outputs.append(x)
        for layer in self.down_layers:
            if isinstance(layer, SSSDSABlock):
                x = layer((x, conditional,diffusion_step_embed))
            else:
                x = layer(x)
            outputs.append(x)

        # Center block
        for layer in self.cent_layers:
            if isinstance(layer, SSSDSABlock):
                x = layer((x,conditional,diffusion_step_embed))
            else:
                x = layer(x)
        x = x + outputs.pop() # add a skip connection to the last output of the down block

        # Up blocks
        for block in self.up_layers:
            if self.use_unet:
                for layer in block:
                    if isinstance(layer, SSSDSABlock):
                        x = layer((x,conditional,diffusion_step_embed))
                    else:
                        x = layer(x)
                    x = x + outputs.pop() # skip connection
            else:
                for layer in block:
                    if isinstance(layer, SSSDSABlock):
                        x = layer((x,conditional,diffusion_step_embed))
                    else:
                        x = layer(x)
                    if isinstance(layer, UpPool):
                        # Before modeling layer in the block
                        x = x + outputs.pop()
                        outputs.append(x)
                x = x + outputs.pop() # add a skip connection from the input of the modeling part of this up block

        return self.norm(x)

    def calc_diffusion_step_embedding(self, diffusion_steps, diffusion_step_embed_dim_in):
        """Embed a diffusion step $t$ into a higher dimensional space
        E.g. the embedding vector in the 128-dimensional space is
        [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]
        Args:
            diffusion_steps: tf.Tensor, diffusion steps for batch data
            diffusion_step_embed_dim_in: int, dimensionality of the
                                    embedding space for discrete diffusion steps
        Returns:
            tf.Tensor, [batch, diffusion_step_embed_dim_in], embedding vectors.
        """
        assert not diffusion_step_embed_dim_in % 2
        half_dim = diffusion_step_embed_dim_in // 2
        exp = tf.math.exp(- tf.range(0., half_dim) * tf.math.log(10000.) / (half_dim - 1))
        # _embed = exp[None] * tf.cast(diffusion_steps[:, None], tf.float32)
        _embed = diffusion_steps * exp
        return tf.concat([tf.sin(_embed), tf.cos(_embed)], axis=-1)


############################ Block for SSSDSA
class SSSDSABlock(tf.keras.Model):
    """WaveNet Block.
    """
    def __init__(self, res_channels,
                 layer,
                 dropout,
                 in_channels,
                 stride
                 ):

        super(SSSDSABlock, self).__init__()
        self.res_channels = res_channels
        self.layer = layer
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.fc_t = tf.keras.layers.Dense(self.res_channels)

        self.cond_conv = DilatedConv(in_channels=2*in_channels,
                                     out_channels=self.res_channels,
                                   kernel_size=stride, stride=stride)

        self.dropout = tf.keras.layers.Dropout(dropout) \
            if dropout > 0 else Identity()

    def call(self, input_data):
        """Pass Block.
        Args:
            input_data, Tuple,
                x: tf.Tensor, [B, T, C(=channels)], input tensor.
                cond: tf.Tensor, [B, T, 2C], embedding tensor for noise schedules.
                diffusion_step_embed: tf.Tensor, [B, T], conditions.
        Returns:
            residual: tf.Tensor, [B, T, C], output tensor for residual connection.
            skip: tf.Tensor, [B, T, C], output tensor for skip connection.
        """
        signal, cond, diffusion_step_embed = input_data
        B, L, C = signal.shape # [B, T, K]
        assert C == self.res_channels
        if C == 256:
            g = 1

        part_t = self.fc_t(diffusion_step_embed)
        h = signal + part_t[:, None]

        h = self.norm(h)
        h = self.layer(h)

        assert cond is not None
        cond = self.cond_conv(cond)
        h += cond

        # Dropout on the output of the layer
        h = self.dropout(h)

        return h + signal


############################ FeedForward Block
class FFBlock(tf.keras.Model):
    """Feed-forward block.
    """
    def __init__(self, d_model, expand=2, dropout=0.0
                 ):
        ''' Initializer
        Args:
        d_model: dimension of input
        expand: expansion factor for inverted bottleneck
        dropout: dropout rate
        '''
        super(FFBlock, self).__init__()

        input_linear = tf.keras.layers.Dense(d_model*expand,
                                                       activation='gelu')
        dropout = tf.keras.layers.Dropout(dropout) \
            if dropout > 0 else Identity()

        output_linear = tf.keras.layers.Dense(d_model)

        self.feedforward = [input_linear,
            dropout,
            output_linear]

    def call(self, x):
        """Pass Block.
        Args:
            x: tf.Tensor, [B, T, C(=channels)], input tensor.
        Returns:
            y: tf.Tensor, [B, T, C], output tensor.
        """
        for layer in self.feedforward:
            x = layer(x)

        return x


# class DilatedConv(tf.keras.layers.Layer):
#     """Custom implementation of dilated convolution 1D
#     because of the issue https://github.com/tensorflow/tensorflow/issues/26797.
#     """
#     def __init__(self,
#                  out_channels,
#                  kernel_size=3,
#                  dilation_rate=1,
#                  stride=1):
#         """Initializer.
#         Args:
#             out_channels: int, output channels.
#             kernel_size: int, size of the kernel.
#             dilation_rate: int, dilation rate.
#         """
#         super(DilatedConv, self).__init__()
#         self.stride = stride
#         self.dilations = dilation_rate
#         self.padding = dilation_rate * (kernel_size - 1) // 2
#         self.out_chn = out_channels
#         self.kernel_size = kernel_size
#
#     def get_config(self):
#         config = super(DilatedConv, self).get_config()
#         config.update({
#             'out_channels': self.out_chn,
#             'kernel_size': self.kernel_size,
#             'dilation_rate': self.dilations,
#             'stride': self.stride,
#         })
#         return config
#
#     def build(self, input_shape):
#
#         self.kernel = self.add_weight(name='kernel',
#                             shape=[self.kernel_size, input_shape[-1], self.out_chn],
#                           initializer=tf.keras.initializers.HeNormal(),
#                           trainable=True)
#         self.bias = self.add_weight(name='bias',
#                             shape=[1, 1, self.out_chn],
#                           initializer=tf.keras.initializers.Zeros(),
#                                     trainable=True)
#
#     def call(self, inputs):
#         """Pass to dilated convolution 1d.
#         Args:
#             inputs: tf.Tensor, [B, T, Cin], input tensor.
#         Returns:
#             outputs: tf.Tensor, [B, T', Cout], output tensor.
#         """
#         conv = tf.nn.conv1d(
#             tf.pad(inputs, self.padding, "CONSTANT"),
#             self.kernel, stride=self.stride, padding='valid', dilations=self.dilations)
#         return conv + self.bias

class DilatedConv(tf.keras.Model):
    """Custom implementation of dilated convolution 1D
    because of the issue https://github.com/tensorflow/tensorflow/issues/26797.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 dilation_rate=1,
                 stride=1):
        """Initializer.
        Args:
            out_channels: int, output channels.
            kernel_size: int, size of the kernel.
            dilation_rate: int, dilation rate.
        """
        super(DilatedConv, self).__init__()
        self.stride = stride
        self.dilations = dilation_rate
        self.padding = dilation_rate * (kernel_size - 1) // 2
        self.out_chn = out_channels
        self.kernel_size = kernel_size
        init = tf.keras.initializers.HeNormal()
        self.kernel = tf.Variable(
            init([kernel_size, in_channels, out_channels], dtype=tf.float32),
            trainable=True)
        self.bias = tf.Variable(
            tf.zeros([1, 1, out_channels], dtype=tf.float32),
            trainable=True)

    def call(self, inputs):
        """Pass to dilated convolution 1d.
        Args:
            inputs: tf.Tensor, [B, T, Cin], input tensor.
        Returns:
            outputs: tf.Tensor, [B, T', Cout], output tensor.
        """
        assert inputs._rank() == 3
        padding = tf.constant([[0, 0], [self.padding, self.padding], [0, 0]],
                              dtype=tf.int32)
        conv = tf.nn.conv1d(
            tf.pad(inputs, padding, "CONSTANT"),
            self.kernel, stride=self.stride, padding='VALID',
            dilations=self.dilations)
        return conv + self.bias


#   #   #   #   #   #   #   #
class DownPool(tf.keras.Model):
    def __init__(self, d_input, expand, pool):
        super().__init__()
        self.d_output = d_input * expand
        self.pool = pool

        self.linear = WeightNormalization(
            tf.keras.layers.Dense(self.d_output))

    def call(self, x):
        x = rearrange(x, '... (l s) h -> ... l (h s)', s=self.pool)
        x = self.linear(x)
        return x


class UpPool(tf.keras.Model):
    def __init__(self, d_input, expand, pool, causal=True):
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool
        self.causal = causal

        self.linear = WeightNormalization(
            tf.keras.layers.Dense(self.d_output * pool))

    def call(self, x):

        x = self.linear(x)
        if (self.causal):
            x = tf.pad(x[..., :-1], (1, 0))  # Shift to ensure causality
        x = rearrange(x, '... l (h s) -> ... (l s) h', s=self.pool)

        return x