import tensorflow as tf
from einops import rearrange
import tensorflow_models as tfm
from .S4Model import S4Block


############################# CDSI with S4layer
class CSDIS4Block(tf.keras.Model):
    """WaveNet Block.
    """
    def __init__(self, res_channels,
                 s4_len,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(CSDIS4Block, self).__init__()

        self.res_channels = res_channels

        self.fc_t = tf.keras.layers.Dense(self.res_channels)

        self.feature_layer = tfm.nlp.models.TransformerEncoder( num_layers=1,
                                    num_attention_heads=8,
                                    intermediate_size=64,
                                    activation='gelu',
                                    norm_first=False,
                                     use_bias=True,
                                    norm_epsilon=1e-05,
                                    intermediate_dropout=0.1)

        self.time_layer = S4Block(features= res_channels,
                           lmax=s4_len,
                           N=s4_d_state,
                           dropout=s4_dropout,
                           bidirectional=s4_bidirectional,
                           layer_norm=s4_layernorm,
                                  postact='gelu')

        self.mid_projection = tf.keras.layers.Conv1D(2 * self.res_channels,  kernel_size=1,
                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                              bias_initializer=tf.keras.initializers.HeNormal())

        self.cond_conv = tf.keras.layers.Conv1D(2 * self.res_channels,
                                kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(),
                                             bias_initializer=tf.keras.initializers.HeNormal())

        self.res_conv = tf.keras.layers.Conv1D(2*self.res_channels,
                                kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(),
                                            bias_initializer=tf.keras.initializers.HeNormal())

    def call(self, input_data):
        """Pass Block.
        Args:
            input_data, Tuple,
                signal: tf.Tensor, [B, T, K, C(=channels)], input tensor.
                cond: tf.Tensor, [B, T, K, 1+time_emb+feature_emb], conditional side information
                diffusion_step_embed: tf.Tensor, [B, T], conditions.
        Returns:
            residual: tf.Tensor, [B, T, K, C], output tensor for residual connection.
            skip: tf.Tensor, [B, T, K, C], output tensor for skip connection.
        """
        signal, cond, diffusion_step_embed = input_data
        base_shape = signal.shape
        B, L, K, C = base_shape[0], base_shape[1], \
            base_shape[2], base_shape[3]

        part_t = self.fc_t(diffusion_step_embed)[:, None]  # (B, 1, C)
        h = rearrange(signal, 'b l k c -> b (l k) c', l=L, k=K) + part_t

        h = self.forward_time(h, (B, L, K, C))
        h = self.forward_feature(h, (B, L, K, C))
        h = self.mid_projection(h)  # (B,K*L, 2C)

        assert cond is not None
        cond = self.cond_conv(rearrange(cond, 'b l k c -> b (l k) c', l=L, k=K))
        h = h + cond

        out = tf.math.tanh(h[..., :self.res_channels]) * \
              tf.math.sigmoid(h[..., self.res_channels:])
        res = self.res_conv(out)

        residual, skip = res[..., :self.res_channels], res[..., self.res_channels:]

        output = (signal + rearrange(residual, 'b (l k) c -> b l k c',
                                     l=L, k=K)) * tf.math.sqrt(0.5)

        return output, rearrange(skip, 'b (l k) c -> b l k c',
                                 l=L, k=K)  # normalize for training stability

    def forward_time(self, y, base_shape):
        """Time S4 Layer.
        Args:
            y: tf.Tensor, [B, T*K, C(=channels)], input tensor.
            base_shape: [B, T, K, C], original shape
        Returns:
            y: tf.Tensor, [B, T, K, C], output tensor
        """
        B, L, K, C = base_shape
        if L == 1:
            return y
        y = self.time_layer(rearrange(y, 'b (l k) c -> (b k) l c', l=L, k=K))
        y = rearrange(y, '(b k) l c -> b l k c', b=B, k=K)
        return y

    def forward_feature(self, y, base_shape):
        """Feature Transformer Layer.
        Args:
            y: tf.Tensor, [B, T*K, C(=channels)], input tensor.
            base_shape: [B, T, K, C], original shape
        Returns:
            y: tf.Tensor, [B, T, K, C], output tensor
        """
        B, L, K, C = base_shape
        if K == 1:
            return y
        y = self.feature_layer(rearrange(y, 'b l k c -> (b l) k c'))
        y = rearrange(y, '(b l) k c -> b (l k) c', b=B, l=L)
        return y


########################## Base CSDI
class CSDIBlock(tf.keras.Model):
    """WaveNet Block.
    """
    def __init__(self, res_channels):
        super(CSDIBlock, self).__init__()
        self.res_channels = res_channels

        self.fc_t = tf.keras.layers.Dense(self.res_channels)

        # encoder_inputs: A tensor with shape `(batch_size, input_length,
        #         hidden_size)`.
        self.time_layer = tfm.nlp.models.TransformerEncoder(num_layers=1,
                                    num_attention_heads=8,
                                    intermediate_size=64,
                                    activation='gelu',
                                    norm_first=False,
                                     use_bias=True,
                                    norm_epsilon=1e-05,
                                    intermediate_dropout=0.1)

        self.feature_layer = tfm.nlp.models.TransformerEncoder( num_layers=1,
                                    num_attention_heads=8,
                                    intermediate_size=64,
                                    activation='gelu',
                                    norm_first=False,
                                    use_bias=True,
                                    norm_epsilon=1e-05,
                                    intermediate_dropout=0.1)

        self.mid_projection = tf.keras.layers.Conv1D(2 * self.res_channels,  kernel_size=1,
                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                              bias_initializer=tf.keras.initializers.HeNormal())

        self.cond_conv = tf.keras.layers.Conv1D(2 * self.res_channels,
                                kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(),
                                             bias_initializer=tf.keras.initializers.HeNormal())

        self.res_conv = tf.keras.layers.Conv1D(2*self.res_channels,
                                kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(),
                                            bias_initializer=tf.keras.initializers.HeNormal())

    def call(self, input_data):
        """Pass Block.
        Args:
            input_data, Tuple,
                signal: tf.Tensor, [B, T, K, C(=channels)], input tensor.
                cond: tf.Tensor, [B, T, K, 1+time_emb+feature_emb], conditional side information
                diffusion_step_embed: tf.Tensor, [B, T], conditions.
        Returns:
            residual: tf.Tensor, [B, T, K, C], output tensor for residual connection.
            skip: tf.Tensor, [B, T, K, C], output tensor for skip connection.
        """
        signal, cond, diffusion_step_embed = input_data
        base_shape = signal.shape
        B, L, K, C = base_shape[0], base_shape[1], \
            base_shape[2], base_shape[3]

        part_t = self.fc_t(diffusion_step_embed)[:, None] # (B, 1, C)
        h = rearrange(signal, 'b l k c -> b (l k) c', l=L, k=K) + part_t

        h = self.forward_time(h, (B, L, K, C))
        h = self.forward_feature(h, (B, L, K, C))
        h = self.mid_projection(h)  # (B,K*L, 2C)

        assert cond is not None
        cond = self.cond_conv(rearrange(cond, 'b l k c -> b (l k) c', l=L, k=K))
        h = h + cond

        out = tf.math.tanh(h[..., :self.res_channels]) * \
              tf.math.sigmoid(h[..., self.res_channels:])
        res = self.res_conv(out)

        residual, skip = res[..., :self.res_channels], res[..., self.res_channels:]

        output = (signal + rearrange(residual, 'b (l k) c -> b l k c',
                  l=L, k=K)) / tf.math.sqrt(2.)

        return output, rearrange(skip, 'b (l k) c -> b l k c',
                  l=L, k=K)  # normalize for training stability

    def forward_time(self, y, base_shape):
        """Time Transformer Layer.
        Args:
            y: tf.Tensor, [B, T*K, C(=channels)], input tensor.
            base_shape: [B, T, K, C], original shape
        Returns:
            y: tf.Tensor, [B, T, K, C], output tensor
        """
        B, L, K, C = base_shape
        if L == 1:
            return y
        y = self.time_layer(rearrange(y, 'b (l k) c -> (b k) l c', l=L, k=K))
        y = rearrange(y, '(b k) l c -> b l k c', b=B, k=K)
        return y

    def forward_feature(self, y, base_shape):
        """Feature Transformer Layer.
        Args:
            y: tf.Tensor, [B, T*K, C(=channels)], input tensor.
            base_shape: [B, T, K, C], original shape
        Returns:
            y: tf.Tensor, [B, T, K, C], output tensor
        """
        B, L, K, C = base_shape
        if K == 1:
            return y
        y = self.feature_layer(rearrange(y, 'b l k c -> (b l) k c'))
        y = rearrange(y, '(b l) k c -> b (l k) c', b=B, l=L)
        return y


################### Main CDSI structure
class CSDIStruct(tf.keras.Model):
    """WaveNet structure.
    """
    def __init__(self, config):
        super(CSDIStruct, self).__init__()
        res_channels = config['res_channels']
        num_res_layers = config['num_res_layers']
        diffusion_step_embed_dim_in = config['diffusion_step_embed_dim_in']
        diffusion_step_embed_dim_mid = config['diffusion_step_embed_dim_mid']
        diffusion_step_embed_dim_out = config['diffusion_step_embed_dim_out']
        if config['with_s4']:
            s4_d_state = config['s4_d_state']
            s4_dropout = config['s4_dropout']
            s4_bidirectional = config['s4_bidirectional']
            s4_layernorm = config['s4_layernorm']
            s4_len = config['s4_len']
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        # for diffusion time embedding
        self.fc_t1 = tf.keras.layers.Dense(diffusion_step_embed_dim_mid)
        self.fc_t2 = tf.keras.layers.Dense(diffusion_step_embed_dim_out)

        if config['with_s4']:
            self.residual_blocks = [CSDIS4Block(res_channels,
                                                s4_len=s4_len,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm) for i in range(self.num_res_layers) ]
        else:
            self.residual_blocks = [CSDIBlock(res_channels) for i in range(self.num_res_layers) ]

    def call(self, input_data):
        """Forwad of CDSI Structure
        Args:
            input_data: tuple,
                noise: tf.Tensor, [B, T, K, res_channels], multivariate time series feature map
                conditional: tf.Tensor, [B, T, K, 1+time_emb+feature_emb], conditional side information
                diffusion_steps: tf.Tensor, [B, 1], diffusion time step
        Returns:
                y: tf.Tensor, [B, T*K, 1], predicted output.
        """
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = self.calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = tf.keras.activations.swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = tf.keras.activations.swish(self.fc_t2(diffusion_step_embed))
        x = rearrange(noise, 'b (l k) c -> b l k c',
                      l=conditional.shape[1], k=conditional.shape[2])
        skip = []
        for layer in self.residual_blocks:
            x, skip_n = layer((x, conditional, diffusion_step_embed))
            skip.append(skip_n)

        skip = tf.reduce_sum(skip, axis=0) * tf.math.sqrt(1.0 / self.num_res_layers)

        return rearrange(skip, 'b l k c -> b (l k) c')

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


