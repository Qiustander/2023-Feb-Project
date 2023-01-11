import json
import tqdm
import os
import os.path as pth
import sys
from glob import glob
sys.path.append(os.getcwd() + '/..')
import wandb
import tensorflow as tf

from typing import Tuple, List, Dict
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

class Trainer:
    '''
    Train Class
    '''
    def __init__(self, config):
        """Initializer.
        Args:
            config: Config, unified configurations.
        """
        self.strategy = tf.distribute.MirroredStrategy() # define multiple-GPU strategy
        self.config = config
        self.train_config = config["train_config"]  # training parameters
        self.dataset_config = config["dataset_config"]  # to load trainset
        self.diffusion_config = config["diffusion_config"]  # basic diffusion hyperparameter
        self.log_config = config["log_config"]    # basic log configuration settings
        self.diffusion_hyperparams = calc_diffusion_hyperparams(
                **self.diffusion_config)
        self.masking = self.train_config['masking']
        self.train_dataset, self.test_dataset = self.dataset_load(self.dataset_config)

        self.model, self.current_step, self.ckpt_path = self.load_model()    # Model Initialization

        self.loss_fun = self.define_dist_loss()

    def define_dist_loss(self):
        with self.strategy.scope():
            loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

            def cal_loss(pred, real):
                per_example_loss = loss_object(pred, real)
                return tf.nn.compute_average_loss(per_example_loss,
                                  global_batch_size=self.B*self.strategy.num_replicas_in_sync)

        return cal_loss

    def load_model(self) -> Tuple[tf.keras.Model, int, str]:
        """ Load model and current training step
        Return:
            model (tf.Model): the network for diffusion
            ckpt_iter (str or int): current training iteration
            ckpt_path (str): checkpoint path
        """
        # A model, an optimizer, and a checkpoint must be created under `strategy.scope`.
        with self.strategy.scope():
            # Note: default eps in tf is 1e-7, in pytorch is 1e-8
            self.optim = tf.keras.optimizers.Adam(
                learning_rate=self.train_config['learning_rate'], epsilon=1e-08)

        # define model
        model_config = self.config['model_config']
        model_config['out_channels'] = self.K
        if self.train_config['use_model'] == 0:
            from model import SSSDSA
            self.log_config['model_name'] = "SSSDSA"
            with self.strategy.scope():
                net = SSSDSA(model_config)
        elif self.train_config['use_model'] == 1:
            model_config['s4_len'] = self.L
            from model import SSSDS4
            self.log_config['model_name'] = "SSSDS4"
            with self.strategy.scope():
                net = SSSDS4(model_config)
        else:
            raise Exception('Model chosen not available.')

        ckpt_path = pth.join(self.log_config['ckpt_path'], self.log_config['model_name'],
                                 self.dataset_config['dataset_name'])
        if not pth.exists(ckpt_path):
            os.makedirs(ckpt_path)

        with open(os.path.join(ckpt_path, 'current_config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)

        kwag = {'model': net,
                'optim': self.optim}
        with self.strategy.scope():
            self.ckpt = tf.train.Checkpoint(**kwag)

        try:
            model_chosen = sorted(glob(ckpt_path + '/*.index', recursive=True))[-1][:-6]
            print('[*] load {} checkpoint at {}. '.format(self.log_config['model_name'],
                                                          pth.basename(model_chosen)[:-2]))
            ckpt_iter = int(pth.basename(model_chosen)[5:-2])

            self.ckpt.restore(model_chosen)
            self.log_initialization(self.log_config, resume_state=True)    # Log Initialization
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
        self.log_initialization(self.log_config, resume_state=False)  # Log Initialization

        return net, ckpt_iter + 1, ckpt_path

    def log_initialization(self, log_config: Dict, resume_state: bool):
        """ Log Configuration. Use Wandb, url: https://wandb.ai/site
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
                   id=log_config['model_name'] + "_{}".format(self.dataset_config['dataset_name']) + log_config['version'],
                   config={"train_config":self.train_config,
                           "diffusion_config":self.diffusion_config},
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
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        data_path = pth.join(dataset_config['data_path'], dataset_config['dataset_name'])
        train_data = np.load(glob(data_path + "/*train*")[0])
        test_data = np.load(glob(data_path + "/*test*")[0])
        shuffle_size_train = train_data.shape[0]

        train_dataset = tf.data.Dataset.\
            from_tensor_slices(train_data).shuffle(shuffle_size_train)\
            .batch(dataset_config['batch_size']*self.strategy.num_replicas_in_sync)
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(self.strategy.num_replicas_in_sync)
        self.B = dataset_config['batch_size']
        self.L = train_data.shape[-2]
        self.K = train_data.shape[-1]

        return (self.strategy.experimental_distribute_dataset(train_dataset),
                self.strategy.experimental_distribute_dataset(test_dataset))

    def compute_loss(self, signal_mask: List, diff_params, gen_missing):
        """Compute loss for noise estimation.
        Args:
            signal_mask: tuple,
                signal (tf.Tensor): [B, T, K], multivariate time series with K features
                mask (tf.Tensor): [B, T, K], mask for imputation target
                loss_mask (tf.Tensor): [B, T, K], mask for loss function
            diff_params (dict): dictionary of diffusion hyperparameters
            gen_missing (int): 0, all sample diffusion. 1, only apply diffusion to missing portions of the signal
        Returns:
            loss (tf.Tensor): MSE-loss between noise and estimation.
        """
        loss_mask = signal_mask[2]
        mask = signal_mask[1]
        # [B, T], [B, T]
        eps, noise = self.diffusion(signal_mask, diff_params, gen_missing)
        # MSE loss
        if gen_missing == 1:
            loss = self.loss_fun(eps*loss_mask, noise*loss_mask)
        elif gen_missing == 0:
            loss = self.loss_fun(eps, noise)
        else:
            raise AssertionError('Define gen_missing case error!')

        return loss

    def genmask(self, train: bool):
        """Generate the mask
        Args:
            train (bool): whether in the training phase
        Returns:
            mask (tf.Tensor): [B, T, K], mask for imputation target
            loss_mask (tf.Tensor): [B, T, K], mask for loss function
        """

        mask = get_mask(self.masking, [self.L, self.K],
                               self.train_config['missing_k'])

        if train:
            mask = tf.convert_to_tensor(mask.repeat(self.B, 0), dtype=tf.float32)
            loss_mask = 1 - mask
            assert self.B == mask.shape[0] == loss_mask.shape[0]
        else:
            mask = tf.convert_to_tensor(mask, dtype=tf.float32)
            loss_mask = 1 - mask
            assert 1 == mask.shape[0] == loss_mask.shape[0]

        assert self.L == mask.shape[1] == loss_mask.shape[1]
        assert self.K == mask.shape[2] == loss_mask.shape[2]

        return mask, loss_mask

    def train(self):
        """Train Network. Use tensorflow2 structure
        """

        max_step = self.train_config['n_iters']
        check_point_step = self.log_config['iters_per_ckpt']
        log_step = self.log_config['iters_per_logging']
        eval_intval = log_step*10
        gen_missing = self.train_config['only_generate_missing']
        step = self.current_step
        loss_test = 999.

        # For loop all the iterations/step
        with tqdm.tqdm(total=max_step, leave=True, desc='Training') as pbar:
            if step:
                pbar.update(step)
            while step < max_step:
                for signal in self.train_dataset:
                    mask, loss_mask = self.genmask(train=True)
                    # Calculate loss
                    loss = self.distributed_train_step(signal=signal, mask=mask,
                                            loss_mask=loss_mask, gen_missing=gen_missing)

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
        gen_missing = self.train_config['only_generate_missing']
        test_num = self.dataset_config['random_test_num']
        choose_idx = np.random.randint(low=0,
                                       high=self.test_dataset.shape[0], size=(test_num))
        loss_test = []
        choose_data = tf.gather(self.test_dataset, choose_idx)

        with tqdm.tqdm(total=test_num, leave=False, desc='Testing') as pbar:
            for signal in choose_data:
                mask, loss_mask = self.genmask(train=False)
                loss_test.append(self.distributed_eval_step(signal=signal, mask=mask,
                                            loss_mask=loss_mask, gen_missing=gen_missing).numpy()
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
        self.ckpt.save(pth.join(self.ckpt_path, "step_{:06d}".format(step)))

        return

    def diffusion(self, signal_mask: List, diff_params, only_generate_missing, eps=None):
        """Trans to next state with diffusion process.
        Args:
            signal_mask: list,
                signal: tf.Tensor, [B, T, K], multivariate time series with K features
                mask: tf.Tensor, [B, T, K], mask for imputation target
                loss_mask: tf.Tensor, [B, T, K], mask for loss function
            diff_params: dict, dictionary of diffusion hyperparameters
            only_generate_missing:  int, 0:all sample diffusion.
                            1: only apply diffusion to missing portions of the signal
            eps: Optional[tf.Tensor: [B, T, K]], noise.
        Return:
            tuple,
                noised: tf.Tensor, [B, T], noised signal.
                eps: tf.Tensor, [B, T], noise.
        """
        assert only_generate_missing == 1 or only_generate_missing == 0
        assert len(signal_mask) == 3

        signal = signal_mask[0]
        cond = signal_mask[0]
        mask = signal_mask[1]
        loss_mask = signal_mask[2]

        B, L, C = signal.shape[0], self.L, self.K  # B is batchsize, C=1, L is signal length
        _dh = diff_params
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
        timesteps = tf.random.uniform(
            shape=[B, 1, 1], minval=0, maxval=T, dtype=tf.int32) # [B, 1], randomly sample diffusion steps from 1~T

        if eps is None:
            eps = tf.random.normal(tf.shape(signal))    # random noise

        if only_generate_missing == 1:
            eps = signal * mask + eps * loss_mask

        extracted_alpha = tf.gather(Alpha_bar, timesteps)
        transformed_X = tf.sqrt(extracted_alpha) * signal + tf.sqrt(
            1 - extracted_alpha) * eps  # compute x_t from q(x_t|x_0)
        timesteps = tf.cast(timesteps, tf.float32)

        epsilon_theta = self.model(
            (transformed_X, cond, mask, tf.squeeze(timesteps, axis=-1)))  # predict \epsilon according to \epsilon_\theta

        return epsilon_theta, eps

    @tf.function
    def distributed_train_step(self, signal, mask, loss_mask, gen_missing):
        per_replica_losses = self.strategy.run(self.train_batch, args=(signal, mask, loss_mask, gen_missing))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)

    def train_batch(self, signal, mask, loss_mask, gen_missing):
        """Warpped training on a batch using static graph.
        Args:
            signal: tf.Tensor, [B, T, K], multivariate time series with K features
            mask: tf.Tensor, [B, T, K], mask for imputation target
            loss_mask: tf.Tensor, [B, T, K], mask for loss function
            gen_missing:  int, 0:all sample diffusion.
                1: only apply diffusion to missing portions of the signal
        Returns:
            loss (float): average loss function of on a batch
        """
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            loss = self.compute_loss([signal, mask, loss_mask],
                                     self.diffusion_hyperparams, gen_missing)
        grad = tape.gradient(loss, self.model.trainable_variables,
                             unconnected_gradients=tf.UnconnectedGradients.ZERO)

        self.optim.apply_gradients(
            zip(grad, self.model.trainable_variables))

        del grad

        return loss

    @tf.function
    def distributed_eval_step(self, signal, mask, loss_mask, gen_missing):
        per_replica_losses = self.strategy.run(self.eval_batch, args=(signal, mask, loss_mask, gen_missing))

        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=None)

    def eval_batch(self, signal, mask, loss_mask, gen_missing):
        """Warpped testing on a batch using static graph.
        Args:
            signal: tf.Tensor, [B, T, K], multivariate time series with K features
            mask: tf.Tensor, [B, T, K], mask for imputation target
            loss_mask: tf.Tensor, [B, T, K], mask for loss function
            gen_missing:  int, 0:all sample diffusion.
                1: only apply diffusion to missing portions of the signal
        Returns:
            loss (float): average loss function of on a batch
        """
        return self.compute_loss([signal, mask, loss_mask],
                                     self.diffusion_hyperparams, gen_missing)

    def dump(self, input):
        """Dump configurations into serializable dictionary.
        Returns:
            y (dict): dictionary for configuration,
        """
        return {k: vars(v) for k, v in vars(input).items()}

