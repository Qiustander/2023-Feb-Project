import json
from glob import glob
import os.path as pth

import tqdm
from typing import Tuple
from statistics import mean

from utils import *


class Generate:
    """Generate data based on ground truth
    Args:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    data_path (str):                  path to dataset, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
    """
    # Get shared output_directory ready
    def __init__(self, config):

        self.config = config
        self.train_config = config["train_config"]  # training parameters
        self.dataset_config = config["dataset_config"]  # to load trainset
        self.diffusion_config = config["diffusion_config"]  # basic diffusion hyperparameter
        self.log_config = config["log_config"]    # basic log configuration settings
        self.diffusion_hyperparams = calc_diffusion_hyperparams(
                **self.diffusion_config)
        self.num_samples = self.dataset_config['num_samples']

        self.masking = self.train_config['masking']

        self.test_dataset = self.definedataset(self.dataset_config)

        self.model, self.out_dir, self.ckpt_path = self.loadmodel()

        self.generate()

    def definedataset(self, dataset_config):
        """Load Pre-processed Dataset from a Directory
        Args:
            dataset_config: Dict
        Return:
            Trainset, Testset: tf.keras.dataloader
        """
        data_path = pth.join(dataset_config['data_path'], dataset_config['dataset_name'])
        test_data = np.load(glob(data_path + "/*test*")[0])

        choose_indx = np.linspace(0, test_data.shape[0] - 1, num=2*dataset_config['random_test_num']).astype(int)
        test_data = np.array(np.split(test_data[choose_indx],
                             len(choose_indx)//dataset_config['random_test_num']*2, 0))

        test_dataset = tf.convert_to_tensor(test_data)
        self.split = test_data.shape[0]
        self.B = test_data.shape[1]
        self.L = test_data.shape[-2]
        self.K = test_data.shape[-1]
        print('Data loaded')
        return test_dataset

    def loadmodel(self) -> Tuple[tf.keras.Model, int, str]:

        # define model
        model_config = self.config['model_config']
        model_config['feature_dim'] = self.K

        if model_config['with_s4']:
            model_config['s4_len'] = self.L
            from model.CSDI import CSDI
            self.log_config['model_name'] = "CSDIS4"
            net = CSDI(model_config)
        else: # s4 false
            from model.CSDI import CSDI
            self.log_config['model_name'] = "CSDI"
            net = CSDI(model_config)

        ckpt_path = pth.join(self.log_config['ckpt_path'], self.log_config['model_name'],
                                 self.dataset_config['dataset_name'])

        try:
            model_chosen = sorted(glob(ckpt_path + '/*.index', recursive=True))[-1][:-6]
            print('[*] load {} checkpoint at {}. '.format(self.log_config['model_name'],
                                                          pth.basename(model_chosen)[:-2]))
            kwag = {'model': net}
            ckpt = tf.train.Checkpoint(**kwag)
            ckpt.restore(model_chosen).expect_partial()
        except:
            raise AssertionError('No valid checkpoint model found.')

        out_dir = pth.join(self.log_config['output_dir'],
                           self.log_config['model_name'],
                           self.dataset_config['dataset_name'])
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            os.chmod(out_dir, 0o775)
        print("output directory", out_dir, flush=True)

        return net, out_dir, ckpt_path

    def genmask(self, signal: tf.Tensor):
        """Generate the mask
            Returns:
                mask (tf.Tensor): mask for imputation
        """

        observed_values, observed_masks, gt_masks = \
            get_mask_CSDI(masking_type=self.masking,
                           data=signal, missing_ratio=self.train_config['missing_k'],
                           k_segments=5)
        return tf.convert_to_tensor(observed_values),\
            tf.convert_to_tensor(observed_masks), tf.convert_to_tensor(gt_masks)

    def generate(self):
        all_RMSE = []
        all_MAE = []
        gen_all_samples = []
        gen_all_mask = []

        with tqdm.tqdm(total=self.split, leave=False, desc='Sampling') as pbar:
            for i, signal in enumerate(self.test_dataset):

                obs_signal, _, gt_mask = self.genmask(signal=signal)
                batch_all_samples = []
                for idx in range(self.num_samples):
                    generated_series = self.sampling(self.model, obs_signal, gt_mask,
                                                     self.diffusion_hyperparams)
                    batch_all_samples.append(generated_series.numpy())

                if not isinstance(gen_all_samples, np.ndarray):
                    gen_all_samples = np.stack(batch_all_samples, axis=1)
                else:
                    gen_all_samples = np.concatenate([gen_all_samples,
                                                       np.stack(batch_all_samples, axis=1)], axis=0)

                gen_all_mask.append(gt_mask)
                pbar.update()

        outfile = 'original.npy'
        new_out = pth.join(self.out_dir, outfile)
        save_signal = np.concatenate([self.test_dataset.numpy()[idx]
                                      for idx in range(self.split)])
        np.save(new_out, save_signal)

        outfile = 'mask.npy'
        new_out = pth.join(self.out_dir, outfile)
        save_mask = np.concatenate(gen_all_mask, axis=0)
        np.save(new_out, save_mask)

        outfile = 'imputation.npy'
        new_out = pth.join(self.out_dir, outfile)
        np.save(new_out, gen_all_samples)

        gen_median = np.median(gen_all_samples, axis=1)
        for batch in range(save_mask.shape[0]):
            comp_mask = save_mask[batch]
            comp_original = save_signal[batch, :]
            comp_gen = gen_median[batch]
            rmse = np.sum(((comp_gen - comp_original) * (1 - comp_mask)) ** 2) / np.sum(1 - comp_mask)
            all_RMSE.append(rmse)
            mae = np.sum(np.abs(((comp_gen - comp_original) * (1 - comp_mask)))) / np.sum(1 - comp_mask)
            all_MAE.append(mae)

        CRPS = calc_quantile_CRPS(
            save_signal, gen_all_samples, 1 - save_mask)

        print('Total RMSE:', np.sqrt(mean(all_RMSE)))
        print('Total MAE:', mean(all_MAE))
        print('Total CRPS:', CRPS)

    # @tf.function
    def sampling(self, net, signal, gt_mask, diff_hyperms
                 ):
        """ Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

        Parameters:
        net (torch network):            the neural-net model
        size (tuple):                   size of tensor to be generated,
                                        usually is (number of audios to generate, channels=1, length of audio)
        diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                        note, the tensors need to be cuda tensors

        Returns:
        the generated audio(s) in torch.tensor, shape=size
        """

        T, Alpha, Alpha_bar, Sigma = diff_hyperms["T"], \
            diff_hyperms["Alpha"], diff_hyperms["Alpha_bar"], diff_hyperms["Sigma"]

        x = tf.random.normal(tf.shape(signal))
        obser_tp = tf.range(signal.shape[1])

        for t in tf.range(T - 1, -1, -1):
            diffusion_steps = tf.cast((t * tf.fill([self.B, 1], 1)),
                                      tf.float32)# use the corresponding reverse step
            total_input = tf.stack([gt_mask * signal,
                                     (1 - gt_mask) * x], axis=-1)  # (B,L,K,2)
            epsilon_theta = net((total_input, obser_tp, gt_mask, diffusion_steps))  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            x = (x - (1 - Alpha[t]) / tf.sqrt(1 - Alpha_bar[t]) * epsilon_theta) \
                / tf.sqrt(Alpha[t])
            if t > 0:
                x += Sigma[t] * tf.random.normal(tf.shape(signal))  # add the variance term to x_{t-1}

        return x

if __name__ == "__main__":
    dataset = "Mujoco"
    config = json.load(open('config/config_CSDI.json'))
    config['dataset_config']['dataset_name'] = dataset
    inf_class = Generate(config)
