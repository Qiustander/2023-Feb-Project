import tensorflow as tf
from typing import Tuple
from .CSDIStruct import CSDIStruct
import numpy as np
from einops import rearrange


class CSDI(tf.keras.Model):
    """CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation
    Tashiro et al., 2021.
    """
    def __init__(self, config):
        """Initializer.
        Args:
            parameters from model configuration.
        """
        super(CSDI, self).__init__()

        self.timeemb = config['timeemb']
        featureemb = config['featureemb']
        res_channels = config['res_channels']
        feature_dim = config['feature_dim']

        self.init_conv = tf.keras.layers.Conv1D(res_channels, kernel_size=1, activation='relu',
                                kernel_initializer=tf.keras.initializers.HeNormal())

        self.residual_layer = CSDIStruct(config)
        self.final_conv = [
            tf.keras.layers.Conv1D(res_channels, kernel_size=1, activation='relu',
                                   kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.Conv1D(1, kernel_size=1,
                                kernel_initializer=tf.keras.initializers.Zeros())
        ]
        self.embed_layer = tf.keras.layers.Embedding(input_dim=feature_dim,
                                                     output_dim=featureemb)

    def call(self, input_data: Tuple):
        """Forwad of CSDI
        Args:
            input_data: tuple,
                noise: tf.Tensor, [B, T, K, 2], stacked of original time series & noised time series at t step
                obser_tp: tf.Tensor, [T,], observed time points
                mask: tf.Tensor, [B, T, K], mask for imputation target
                diffusion_steps: tf.Tensor, [B, 1], diffusion time step
        Returns:
                y: tf.Tensor, [B, T, K], predicted output.
        """

        noise, obser_tp, gt_mask, diffusion_steps = input_data

        conditional = self.get_side_info(obser_tp, gt_mask)

        x = rearrange(noise, 'b l k c -> b (l k) c')
        x = self.init_conv(x) # b (l k) c

        x = self.residual_layer((x, conditional, diffusion_steps)) # b (l k) c
        for layer in self.final_conv:
            x = layer(x) # b (l k) 1

        return rearrange(tf.squeeze(x, axis=-1), 'b (l k) -> b l k',
                         l=noise.shape[1], k=noise.shape[2])

    def get_side_info(self, observed_tp, cond_mask):
        """Get side information (conditional info) for the network
        Args:
            observed_tp: tf.Tensor, [T,], observed time points
            cond_mask: tf.Tensor, [B, T, K], mask for imputation target
        Returns:
            y: tf.Tensor, [B, T, K, 1+time_emb+feature_emb], side information
        """
        B, L, K = cond_mask.shape

        # Eager mode
        # time_embed = self.time_embedding(tf.repeat(observed_tp[None], repeats=B, axis=0) , self.timeemb)  # (B,L,emb_time)
        # time_embed = tf.convert_to_tensor(time_embed, dtype=tf.float32)

        # Static mode
        time_embed = tf.py_function(func=self.time_embedding,
                                    inp=[tf.repeat(observed_tp[None], repeats=B, axis=0), self.timeemb], Tout=tf.float32)  # (B,L,emb_time)
        time_embed.set_shape([B, L, self.timeemb])
        time_embed = tf.repeat(time_embed[:,:,None], repeats=K, axis=2) # (B,L,K, emb_time)
        feature_embed = self.embed_layer(tf.range(K))  # (K,emb_fea)
        feature_embed = tf.tile(feature_embed[None, None], [B, L, 1, 1]) # (B,L,K,emb_fea)

        side_info = tf.concat([time_embed, feature_embed, cond_mask[..., None]], axis=-1)  # (B,L,K,*+1)

        return side_info

    def time_embedding(self, pos, d_model=128):
        """Sinusoidal time embedding.
        Args:
            pos: tf.Tensor, [B, T], input embedding
            d_model: int, embedding length
        Returns:
            y: tf.Tensor, [B, T, time_emb], time embedding
        """
        pe = np.zeros((pos.shape[0], pos.shape[1], d_model))
        position = pos[:,:,None].numpy()
        div_term = 1 / np.power(10000.0, np.arange(0, d_model, 2) / d_model)
        pe[:, :, 0::2] = np.sin(position * div_term)
        pe[:, :, 1::2] = np.cos(position * div_term)
        return pe