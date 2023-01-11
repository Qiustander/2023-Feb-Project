import tensorflow as tf
import json
import tqdm
import os
import os.path as pth
import sys
from glob import glob
sys.path.append(os.getcwd() + '/..')
import wandb

from typing import Tuple, List, Dict
from utils import *

class Trainer:
    '''
    Train Class - for CSDI
    '''
    def __init__(self, config):
        """Initializer.
        Args:
            config: Config, unified configurations.
        """
        self.config = config
        self.train_config = config["train_config"]  # training parameters
        self.dataset_config = config["dataset_config"]  # to load trainset
        self.diffusion_config = config["diffusion_config"]  # basic diffusion hyperparameter
        self.log_config = config["log_config"]    # basic log configuration settings
        # self.loss_func = tf.keras.losses.RootMeanSquaredError()
        self.diffusion_hyperparams = calc_diffusion_hyperparams(
                **self.diffusion_config)
        self.masking = self.train_config['masking']

        # Note: default eps in tf is 1e-7, in pytorch is 1e-8
        self.optim = tf.keras.optimizers.Adam(
            learning_rate=self.train_config['learning_rate'], epsilon=1e-08)

        self.train_dataset, self.test_dataset = self.dataset_load(self.dataset_config)

        self.model, self.current_step, self.ckpt_path = self.load_model()    # Model Initialization

    def load_model(self) -> Tuple[tf.keras.Model, int, str]:
        """ Load model and current training step
        Return:
            model (tf.Model): the network for diffusion
            ckpt_iter (str or int): current training iteration
            ckpt_path (str): checkpoint path
        """

        # define model
        model_config = self.config['model_config']
        model_config['feature_dim'] = self.K
        if model_config['with_s4']:
            model_config['s4_len'] = self.L
            from model import CSDI
            self.log_config['model_name'] = "CSDIS4"
            net = CSDI(model_config)
        else: # s4 false
            from model import CSDI
            self.log_config['model_name'] = "CSDI"
            net = CSDI(model_config)

        ckpt_path = pth.join(self.log_config['ckpt_path'], self.log_config['model_name'],
                                 self.dataset_config['dataset_name'])
        if not pth.exists(ckpt_path):
            os.makedirs(ckpt_path)

        with open(os.path.join(ckpt_path, 'current_config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        try:
            model_chosen = sorted(glob(ckpt_path + '/*.index', recursive=True))[-1][:-6]
            print('[*] load {} checkpoint at {}. '.format(self.log_config['model_name'],
                                                          pth.basename(model_chosen)[:-2]))
            ckpt_iter = int(pth.basename(model_chosen)[5:-2])
            kwag = {'model': net,
                    'optim': self.optim}
            ckpt = tf.train.Checkpoint(**kwag)
            ckpt.restore(model_chosen)
            self.log_initialization(self.log_config, resume_state=True)  # Log Initialization
        except:
            ckpt_iter = -1
            print('No valid checkpoint of model {} found, '
                  'start training from initialization try.'.format(self.log_config['model_name']))

            self.log_initialization(self.log_config, resume_state=False)  # Log Initialization

        return net, ckpt_iter + 1, ckpt_path

    def log_initialization(self, log_config: Dict, resume_state: bool):
        """Log Configuration. Use Wandb, url: https://wandb.ai/site
        Args:
            log_config (dict): dictionary for log configuration
            resume_state (bool): resume pretraining state
        """

        if resume_state:
            resume = 'must'
        else:
            resume = False

        wandb.login()
        wandb.init(project=log_config['project_name'],
                   name=log_config['model_name'] + "_{}".format(self.dataset_config['dataset_name']),
                   id=log_config['model_name'] + "_{}".format(self.dataset_config['dataset_name']) + log_config[
                       'version'],
                   config={"train_config": self.train_config,
                           "diffusion_config": self.diffusion_config},
                   resume=resume)
        if log_config['dry_run']:
            os.environ['WANDB_MODE'] = 'dryrun'

        return

    def dataset_load(self, dataset_config) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load Pre-processed Dataset from a Directory
        Args:
            dataset_config (dict): dataset configuration
        Return:
            Trainset, Testset (tf.Tensor): training and testing dataset
        """
        data_path = pth.join(dataset_config['data_path'], dataset_config['dataset_name'])
        train_data = np.load(glob(data_path + "/*train*")[0])
        test_data = np.load(glob(data_path + "/*test*")[0])
        shuffle_size_train = train_data.shape[0]

        train_dataset = tf.data.Dataset.\
            from_tensor_slices(train_data).shuffle(shuffle_size_train)\
            .batch(dataset_config['batch_size'])
        test_dataset = tf.convert_to_tensor(test_data)
        self.B = dataset_config['batch_size']
        self.L = train_data.shape[-2]
        self.K = train_data.shape[-1]

        self.mask_size = [self.B, self.L, self.K]

        return (train_dataset, test_dataset)

    def compute_loss(self, signal_mask: List, diff_params):
        """Compute loss for noise estimation.
        Args:
            signal_mask: tuple,
                signal (tf.Tensor): [B, T, K], multivariate time series with K features
                obs_mask (tf.Tensor): [B, T, K], mask for observation points
                gt_mask (tf.Tensor): [B, T, K], mask for imputation target
                diff_params (dict): dictionary of diffusion hyperparameters
        Returns:
            loss (tf.Tensor): MSE-loss between noise and estimation.
        """
        obs_mask = signal_mask[1]
        gt_mask = signal_mask[2]
        # [B, T], [B, T]
        epsilon_theta, eps = self.diffusion(signal_mask, diff_params)
        # MSE loss
        target_mask = obs_mask - gt_mask
        residual = (epsilon_theta - eps) * target_mask

        loss = tf.reduce_sum(residual**2)/ (tf.reduce_sum(target_mask)
                                            if tf.reduce_sum(target_mask)>0 else 1.0)
        return loss

    def genmask(self, signal: tf.Tensor, train=True):
        """Generate the mask
        Returns:
            observed_values (tf.Tensor): [B, T, K], [B, T, K], multivariate time series with K features
            observed_masks (tf.Tensor): [B, T, K], mask for observation points
            gt_masks (tf.Tensor): [B, T, K], mmask for imputation target
        """
        miss_ratio = self.train_config['missing_k'] if not train else None
        observed_values, observed_masks, gt_masks = \
            get_mask_CSDI(masking_type=self.masking,
                           data=signal, missing_ratio=miss_ratio,
                           k_segments=5)
        return observed_values, observed_masks, gt_masks

    def train(self):
        """Train Network. Use tensorflow2 structure
        """

        max_step = self.train_config['n_iters']
        check_point_step = self.log_config['iters_per_ckpt']
        log_step = self.log_config['iters_per_logging']
        eval_intval = log_step*10
        step = self.current_step
        loss_test = 999.

        # For loop all the iterations/step
        with tqdm.tqdm(total=max_step, leave=True, desc='Training') as pbar:
            if step:
                pbar.update(step)
            while step < max_step:
                for signal in self.train_dataset:
                    obs_signal, obs_mask, gt_mask = self.genmask(signal=signal)
                    # Calculate loss
                    loss = self.train_batch(signal=obs_signal, obs_mask=obs_mask,
                                            gt_mask=gt_mask)

                    step += 1
                    pbar.update()
                    pbar.set_postfix(
                        {'loss': loss.numpy(),
                         'step': step,
                         'loss_test': loss_test})

                    # logging
                    if not step % log_step:
                        wandb.log({"train_loss": loss.numpy()})
                        if not step % eval_intval:
                            loss_test = self.eval_result()
                            wandb.log({"test_loss": loss_test})

                    # Save_model
                    if step % check_point_step == 0:
                        self.save_model(step)

    def eval_result(self):
        """Compute the loss over the testset.
        Returns:
            loss (float): average loss function of the testing dataset
        """
        test_num = self.dataset_config['random_test_num']
        choose_idx = np.random.randint(low=0,
                                       high=self.test_dataset.shape[0], size=(test_num))
        loss_test = []
        choose_data = tf.gather(self.test_dataset, choose_idx)

        with tqdm.tqdm(total=test_num, leave=False, desc='Testing') as pbar:
            for signal in choose_data:
                signal = signal[None]
                observed_values, observed_masks, gt_masks = self.genmask(signal=signal, train=False)
                loss_test.append(self.eval_batch(signal=observed_values, obs_mask=observed_masks,
                                            gt_mask=gt_masks).numpy()
                    )

                pbar.update()
                pbar.set_postfix(
                    {'test_loss': loss_test[-1]})

        return sum(loss_test) / len(loss_test)

    def save_model(self, step):
        """Save Checkpoint
        Args:
            step (int): current training step.
        """
        model_list = sorted(glob(self.ckpt_path + '/*.index', recursive=True))
        if len(model_list) >= self.log_config['save_model_num']:
            delete_file = glob(self.ckpt_path + '/*{}*'.format(pth.basename(model_list[0])[:12]), recursive=True)
            for del_f in delete_file:
                os.remove(del_f)
        kwag = {'model': self.model,  'optim': self.optim}
        ckpt = tf.train.Checkpoint(**kwag)
        ckpt.save(pth.join(self.ckpt_path, "step_{:06d}".format(step)))

        return

    def diffusion(self, signal_mask: List, diff_params, eps=None):
        """Trans to next state with diffusion process.
        Args:
            signal_mask: list,
                signal (tf.Tensor): [B, T, K], multivariate time series with K features
                obs_mask (tf.Tensor): [B, T, K], mask for observation points
                gt_mask (tf.Tensor): [B, T, K], mask for imputation target
            diff_params: dict, dictionary of diffusion hyperparameters
            eps: Optional[tf.Tensor: [B, T, K]], noise.
        Return:
            tuple,
                noised: tf.Tensor, [B, T, K], noised signal.
                eps: tf.Tensor, [B, T, K], noise.
        """
        #
        # def set_input_to_diffmodel(noisy_data, observed_data, cond_mask):
        #     total_input = tf.stack([(cond_mask * observed_data),
        #                              ((1 - cond_mask) * noisy_data)], axis=-1)  # (B,L,K,2)
        #     return total_input

        assert len(signal_mask) == 3

        signal = signal_mask[0]
        cond_mask = signal_mask[2]

        B, L, C = signal.shape[0], self.L, self.K  # B is batchsize, C=1, L is signal length
        _dh = diff_params
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
        timesteps = tf.random.uniform(
            shape=[B, 1, 1], minval=0, maxval=T, dtype=tf.int32) # [B], randomly sample diffusion steps from 1~T

        if eps is None:
            eps = tf.random.normal(tf.shape(signal))    # random noise

        extracted_alpha = tf.gather(Alpha_bar, timesteps)
        transformed_X = tf.sqrt(extracted_alpha) * signal + tf.sqrt(
            1 - extracted_alpha) * eps  # compute x_t from q(x_t|x_0)
        timesteps = tf.cast(timesteps, tf.float32)
        # total_input = set_input_to_diffmodel(transformed_X, signal, cond_mask) # B, L, K, 2
        total_input = tf.stack([cond_mask * signal,
                                     (1 - cond_mask) * transformed_X], axis=-1) # B, L, K, 2
        obser_tp = tf.range(signal.shape[1])

        epsilon_theta = self.model(
            (total_input, obser_tp, cond_mask, tf.squeeze(timesteps, axis=-1)))  # predict \epsilon according to \epsilon_\theta

        return epsilon_theta, eps

    @tf.function
    def train_batch(self, signal, obs_mask, gt_mask):
        """Warpped training on a batch using static graph.
        Args:
            signal (tf.Tensor): [B, T, K], multivariate time series with K features
            obs_mask (tf.Tensor): [B, T, K], mask for observation points
            gt_mask (tf.Tensor): [B, T, K], mask for imputation target
        Returns:
            loss (float): average loss function of on a batch
        """
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.compute_loss([signal, obs_mask, gt_mask],
                                     self.diffusion_hyperparams)

        grad = tape.gradient(loss, self.model.trainable_variables,
                             unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.optim.apply_gradients(
            zip(grad, self.model.trainable_variables))
        del grad

        return loss

    @tf.function
    def eval_batch(self, signal, obs_mask, gt_mask):
        """Warpped testing on a batch using static graph.
        Args:
            signal (tf.Tensor): [B, T, K], multivariate time series with K features
            obs_mask (tf.Tensor): [B, T, K], mask for observation points
            gt_mask (tf.Tensor): [B, T, K], mask for imputation target
        Returns:
            loss (float): average loss function of on a batch
        """
        return self.compute_loss([signal, obs_mask, gt_mask],
                                     self.diffusion_hyperparams)

    def dump(self, input):
        """Dump configurations into serializable dictionary.
        Returns:
            y (dict): dictionary for configuration,
        """
        return {k: vars(v) for k, v in vars(input).items()}

