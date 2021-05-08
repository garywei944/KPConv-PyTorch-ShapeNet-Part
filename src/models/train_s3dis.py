#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on S3DIS dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os

# Dataset
from src.features.s3dis import *
from torch.utils.data import DataLoader

from src.config.config import Config
from src.models.trainer import ModelTrainer
from src.models.architectures import KPFCNN

import multiprocessing
from ruamel import yaml
import argparse


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class S3DISConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    def __init__(self, config_file=None):
        super().__init__()
        if config_file:
            with open(os.path.join('config', config_file), 'r') as f:
                d = yaml.safe_load(f)
                for k, v in d.items():
                    exec(k + '=v')

    max_epoch = -1  # Just to remove IDE error msg

    # Load the baseline configuration
    with open('config/s3dis_baseline.yml', 'r') as f:
        d = yaml.safe_load(f)
        for k, v in d.items():
            exec(k + '=v')

    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c', '--config', type=str,
                        help='the yml config file in config folder')
    parser.add_argument('-p', '--saving-path', type=str,
                        help='the path to save the results')

    args = parser.parse_args()

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)
    # previous_training_path = 'Log_2020-03-19_19-53-27'
    previous_training_path = ''

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path,
                                 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path,
                                   'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = S3DISConfig(config_file=args.config)
    if previous_training_path:
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None

    # Get path from argument if given
    config.saving_path = args.saving_path

    # Initialize datasets
    training_dataset = S3DISDataset(config, set='training', use_potentials=True)
    test_dataset = S3DISDataset(config, set='validation', use_potentials=True)

    # Initialize samplers
    training_sampler = S3DISSampler(training_dataset)
    test_sampler = S3DISSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=S3DISCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=S3DISCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    # Optional debug functions
    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_upsampling(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPFCNN(config, training_dataset.label_values,
                 training_dataset.ignored_labels)

    debug = False
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(
            param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    # Define a trainer class
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, test_loader, config)

    # print('Forcing exit now')
    # os.kill(os.getpid(), signal.SIGINT)
