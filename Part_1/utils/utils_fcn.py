import tensorflow as tf
import numpy as np
import random
import os

def calc_diffusion_hyperparams(T, beta_0, beta_T, strategy='linear'):
    """Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (tf.tensor, shape=(T, ))
    """
    if strategy == 'linear':
        Beta = np.linspace(beta_0, beta_T, T)  # Linear schedule
    elif strategy == 'quadratic':
        Beta = np.linspace(beta_0 ** 0.5, beta_T ** 0.5, T) ** 2
    else:
        raise AssertionError('Wrong diffusion strategy')
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = np.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    diffusion_hyperparams = {}
    diffusion_hyperparams["T"], diffusion_hyperparams["Beta"], \
        diffusion_hyperparams["Alpha"], diffusion_hyperparams["Alpha_bar"], diffusion_hyperparams["Sigma"] = \
        tf.convert_to_tensor(T), tf.convert_to_tensor(Beta, dtype=tf.float32), \
            tf.convert_to_tensor(Alpha, dtype=tf.float32), tf.convert_to_tensor(Alpha_bar, dtype=tf.float32), \
            tf.convert_to_tensor(Sigma, dtype=tf.float32)
    return diffusion_hyperparams


def get_mask(masking_type, sample_shape, k):

    if masking_type == 'rm':
        """Get mask of random points (missing at random) across channels based on k,
        where k == number of data points. Mask of sample's shape where 0's to be imputed,
        and 1's to preserved as per ts imputers.
        It is more convenient to use Numpy to do the indexing then conver to tf.tensor

        """
        mask = np.ones(sample_shape)
        length_index = np.arange(mask.shape[0])  # lenght of series indexes
        for channel in range(mask.shape[1]):
            perm = np.random.permutation(len(length_index))
            idx = perm[0:k]
            mask[:, channel][idx] = 0

        assert mask.shape[0] == sample_shape[0]
        assert mask.shape[1] == sample_shape[1]

        return np.expand_dims(mask, 0)

    elif masking_type == 'mnr':
        """Get mask of random segments (non-missing at random) across channels based on k,
        where k == number of segments. Mask of sample's shape where 0's to be imputed,
        and 1's to preserved as per ts imputers
        It is more con`venient to use Numpy to do the indexing then conver to tf.tensor
        """
        mask = np.ones(sample_shape)
        length_index = np.arange(mask.shape[0])
        list_of_segments_index = np.split(length_index, length_index // k)
        for channel in range(mask.shape[1]):
            s_nan = random.choice(list_of_segments_index)
            mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

        assert mask.shape[0] == sample_shape[0]
        assert mask.shape[1] == sample_shape[1]

        return np.expand_dims(mask, 0)

    elif masking_type == 'bm':
        """Get mask of same segments (black-out missing) across channels based on k,
        where k == number of segments. Mask of sample's shape where 0's to be imputed,
        and 1's to be preserved as per ts imputers
        It is more convenient to use Numpy to do the indexing then conver to tf.tensor

        """
        mask = np.ones(sample_shape)
        length_index = np.arange(mask.shape[0])
        list_of_segments_index = np.split(length_index, length_index // k)
        s_nan = random.choice(list_of_segments_index)
        for channel in range(mask.shape[1]):
            mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

        assert mask.shape[0] == sample_shape[0]
        assert mask.shape[1] == sample_shape[1]

        return np.expand_dims(mask, 0)
    else:
        raise AssertionError('No masking type!')


def get_mask_CSDI(masking_type, data, missing_ratio=None, k_segments=5):
    # Input data - B, L, K
    if masking_type == 'rm':
        observed_values = data.numpy()
        observed_mask = ~np.isnan(observed_values)

        rand_for_mask = np.random.rand(*observed_mask.shape) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1) # B, L*K
        for i in range(len(observed_mask)): # Loop for Batch
            sample_ratio = np.random.rand() if not missing_ratio else missing_ratio # missing ratio
            num_observed = observed_mask[i].sum()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][np.argpartition(rand_for_mask[i], -num_masked)[-num_masked:]] = -1
        gt_masks = (rand_for_mask > 0).reshape(observed_mask.shape)

        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_mask.astype(np.single)
        gt_masks = gt_masks.astype(np.single)

        return observed_values, observed_masks, gt_masks

    elif masking_type == 'mnr':
        observed_values = np.array(data)
        observed_masks = ~np.isnan(observed_values)
        gt_masks = observed_masks.copy()
        length_index = np.array(range(data.shape[0]))
        list_of_segments_index = np.array_split(length_index, k_segments)

        for channel in range(gt_masks.shape[1]):
            s_nan = random.choice(list_of_segments_index)
            gt_masks[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype("float32")
        gt_masks = gt_masks.astype("float32")

        return observed_values, observed_masks, gt_masks

    elif masking_type == 'bm':
        observed_values = np.array(data)
        observed_masks = ~np.isnan(observed_values)
        gt_masks = observed_masks.copy()
        length_index = np.array(range(data.shape[0]))
        list_of_segments_index = np.array_split(length_index, k_segments)
        s_nan = random.choice(list_of_segments_index)

        for channel in range(gt_masks.shape[1]):
            gt_masks[:, channel][s_nan[0]:s_nan[-1] + 1] = 0

        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype("float32")
        gt_masks = gt_masks.astype("float32")

        return observed_values, observed_masks, gt_masks
    else:
        raise AssertionError('No masking type!')


def _calc_denominator(target, eval_points):
    return np.sum(np.abs(target * eval_points))


def _quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * np.sum(np.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q)))


def calc_quantile_CRPS(target, forecast, eval_points):
    """Calculate the continuous ranked probability score (CRPS) in Python.
    Args:
    Returns:
        np.float, the CRPS loss value
    """

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = _calc_denominator(target, eval_points)
    CRPS = 0
    # for i in range(len(quantiles)):
    #     q_pred = np.quantile(forecast, quantiles[i], axis=0, keepdims=True)
    #     q_loss = _quantile_loss(target, q_pred, quantiles[i], eval_points)
    #     CRPS += q_loss / denom
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(np.quantile(forecast[j : j + 1], quantiles[i], axis=1))
        q_pred = np.concatenate(q_pred, 0)
        q_loss = _quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom

    return CRPS / len(quantiles)

