from typing import Tuple
import tensorflow as tf
from tensorflow_addons.layers import WeightNormalization
from .SSSDStruct import SSSDStuct
from .SSSDStruct import SSSDSA


class SSSD(tf.keras.Model):
    """Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models
    Alcaraz et al., 2022.
    With S4 model imbedded.
    """
    def __init__(self, config):
        """Initializer.
        Args:
            parameters from model configuration.
        """
        super(SSSD, self).__init__()
        self.config = config
        out_channels = config['out_channels']
        res_channels = config['res_channels']
        # skip_channels = config['skip_channels']
        is_SA_struct = config['is_SA_struct']

        self.init_conv = WeightNormalization(tf.keras.layers.Conv1D(res_channels, kernel_size=1, activation='relu',
                                kernel_initializer=tf.keras.initializers.HeNormal(),
                                bias_initializer=tf.keras.initializers.HeNormal()))


        if is_SA_struct:
            self.residual_layer = SSSDSA(self.config)
        else:
            self.residual_layer = SSSDStuct(self.config)
        self.final_conv = [
            WeightNormalization(tf.keras.layers.Conv1D(res_channels, kernel_size=1, activation='relu',
                                   kernel_initializer=tf.keras.initializers.HeNormal(),
                                   bias_initializer=tf.keras.initializers.HeNormal())),
            tf.keras.layers.Conv1D(out_channels, kernel_size=1,
                                kernel_initializer=tf.keras.initializers.Zeros(),
                                   bias_initializer=tf.keras.initializers.Zeros())
        ]

    def call(self, input_data: Tuple):
        """Forwad of SSSD
        Args:
            input_data: tuple,
                noise: tf.Tensor, [B, T, K], noised multivariate time series at t step
                conditional: tf.Tensor, [B, T, K], original multivariate time series
                mask: tf.Tensor, [B, T, K], mask for imputation target
                diffusion_steps: tf.Tensor, [B, 1], diffusion time step
        Returns:
                y: tf.Tensor, [B, T, K], predicted output.
        """
        noise, conditional, mask, diffusion_steps = input_data

        conditional = tf.concat([conditional * mask, mask], axis=-1)

        x = self.init_conv(noise)
        x = self.residual_layer((x, conditional, diffusion_steps))
        for layer in self.final_conv:
            x = layer(x)
        return x

