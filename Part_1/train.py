import json

train_model = 'SSSD'
dataset = "Mujoco"
is_sa = False

if train_model == 'SSSD':
    if is_sa:
        config = json.load(open('config/config_SSSDSA.json'))
    else:
        config = json.load(open('config/config_SSSDS4.json'))
    config['dataset_config']['dataset_name'] = dataset
    if config['dataset_config']['multiple_gpu']:
        from trainer.trainer_mulgpu import Trainer
    else:
        from trainer.trainer import Trainer
elif train_model == 'CSDI':
    config = json.load(open('config/config_CSDI.json'))
    config['dataset_config']['dataset_name'] = dataset
    from trainer.trainer_CSDI import Trainer
else:
    raise AssertionError('No train model')
assert config is not None
"""
Read configuration
"""

'''Notice:
The Conv2D op currently only supports the NHWC tensor format on the CPU. 
Should follow the channel ordering of the tensorflow.
Although keras provide channel_first for image data format performming like Pytorch,
it does not apply for the time series data.
'''
trainer = Trainer(config)
trainer.train()

