import logging
import time
import copy
import datetime
import random
import argparse
import pathlib
import wandb
from random import choice
from string import ascii_uppercase
import os
import torch
import numpy as np
from tensorboardX import SummaryWriter

from utils.torch_metrics import compute_cross_entropy, compute_batch_metrics
from utils.utils import (add_mask_params, save_json, build_optim, count_parameters,
                               count_trainable_parameters, count_untrainable_parameters, str2bool, str2none)
from data.data_loading import create_data_loader
from inference_model.inference_model_utils import load_infer_model
from policy_model.policy_model_utils import (build_policy_model, load_policy_model, save_policy_model,
                                                 compute_scores, compute_backprop_trajectory,
                                                 compute_next_step_inference, get_policy_probs)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def train_epoch(args, epoch, infer_model, model, loader, optimiser, writer):
    """
    Performs a single training epoch.

    :param args: Argument object, containing hyperparameters for model training.
    :param epoch: int, current training epoch.
    :param recon_model: reconstruction model object.
    :param model: policy model object.
    :param loader: training data loader.
    :param optimiser: PyTorch optimizer.
    :param writer: Tensorboard writer.
    :param data_range_dict: dictionary containing the dynamic range of every volume in the training data.
    :return: (float: mean loss of this epoch, float: epoch duration)
    """
    model.train()
    epoch_loss = [0. for _ in range(args.acquisition_steps)]
    report_loss = [0. for _ in range(args.acquisition_steps)]
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(loader)

    cbatch = 0  # Counter for spreading single backprop batch over multiple data loader batches
    for it, data in enumerate(loader):  # Loop over data points
        cbatch += 1
        kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, slice_info = data
        # shape after unsqueeze = batch x channel x columns x rows x complex
        kspace = kspace.unsqueeze(1).to(args.device)
        masked_kspace = masked_kspace.unsqueeze(1).to(args.device)
        mask = mask.unsqueeze(1).to(args.device)
        # shape after unsqueeze = batch x channel x columns x rows
        zf = zf.to(args.device)
        gt = gt.to(args.device)
        label = slice_info['label'].to(args.device)
        # Base inference model forward pass
        if args.use_feature_map:
            feature_map, outputs = infer_model(zf,label)
            image_input = feature_map
        else:
            outputs = infer_model(zf)
            image_input = zf[:, 0, :, :].unsqueeze(1)

        if cbatch == 1:  # Only after backprop is performed
            optimiser.zero_grad()

        action_list = []
        logprob_list = []
        reward_list = []
        for step in range(args.acquisition_steps):  # Loop over acquisition steps
            loss, mask, masked_kspace,  image_input, outputs = compute_backprop_trajectory(args, kspace, masked_kspace, mask, outputs, image_input, label, model, infer_model, step,
                                                                            action_list, logprob_list, reward_list)
            # Loss logging
            epoch_loss[step] += loss.item() / len(loader) * gt.size(0) / args.batch_size
            report_loss[step] += loss.item() / args.report_interval * gt.size(0) / args.batch_size
            writer.add_scalar('TrainLoss_step{}'.format(step), loss.item(), global_step + it)

        # Backprop if we've reached the prerequisite number of dataloader batches
        if cbatch == args.batches_step:
            optimiser.step()
            cbatch = 0

        # Logging: note that loss values mean little, as the Policy Gradient loss is not a true loss.
        if it % args.report_interval == 0:
            if it == 0:
                loss_str = ", ".join(["{}: {:.2f}".format(i + 1, args.report_interval * l * 1e3)
                                      for i, l in enumerate(report_loss)])
            else:
                loss_str = ", ".join(["{}: {:.2f}".format(i + 1, l * 1e3) for i, l in enumerate(report_loss)])
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}], '
                f'Iter = [{it:4d}/{len(loader):4d}], '
                f'Time = {time.perf_counter() - start_iter:.2f}s, '
                f'Avg Loss per step x1e3 = [{loss_str}] ',
            )
            report_loss = [0. for _ in range(args.acquisition_steps)]

        start_iter = time.perf_counter()

    if args.wandb:
        wandb.log({'train_loss_step': {str(key + 1): val for key, val in enumerate(epoch_loss)}}, step=epoch + 1)

    return np.mean(epoch_loss), time.perf_counter() - start_epoch


def evaluate(args, epoch, infer_model, model, loader, writer, partition):
    """
    Evaluates the policy on all slices in a validation or test dataset on the SSIM and PSNR metrics.

    :param args: Argument object, containing hyperparameters for model evaluation.
    :param epoch: int, current training epoch.
    :param recon_model: reconstruction model object.
    :param model: policy model object.
    :param loader: training data loader.
    :param writer: Tensorboard writer.
    :param partition: str, dataset partition to evaluate on ('val' or 'test')
    :param data_range_dict: dictionary containing the dynamic range of every volume in the validation or test data.
    :return: (dict: average SSIMS per time step, dict: average PSNR per time step, float: evaluation duration)
    """
    model.eval()
    cross_entropy, accuracy = 0, 0
    tbs = 0  # data set size counter
    start = time.perf_counter()

    for it, data in enumerate(loader):
        kspace, masked_kspace, mask, zf, gt, gt_mean, gt_std, fname, slice_info = data

        # shape after unsqueeze = batch x channel x columns x rows x complex
        kspace = kspace.unsqueeze(1).to(args.device)
        masked_kspace = masked_kspace.unsqueeze(1).to(args.device)
        mask = mask.unsqueeze(1).to(args.device)

        # shape after unsqueeze = batch x channel x columns x rows
        zf = zf.to(args.device)

        label = slice_info['label'].to(args.device)

        tbs += mask.size(0)
        with torch.no_grad():
            # Base inference model forward pass
            if args.use_feature_map:
                feature_map, outputs = infer_model(zf)
                image_input = feature_map
            else:
                outputs = infer_model(zf)
                image_input = zf[:, 0, :, :].unsqueeze(1)
        # print(outputs)
        init_cross_entropy_val = compute_cross_entropy(outputs, label)
        batch_cross_entropy = [init_cross_entropy_val]
        init_acc_val = compute_batch_metrics(outputs, label)['accuracy']
        batch_accuracy = [init_acc_val]


        for step in range(args.acquisition_steps):
            policy, probs = get_policy_probs(model, image_input, mask)
            if step == 0:
                actions = torch.multinomial(probs.squeeze(1), args.num_test_trajectories, replacement=True)
            else:
                actions = policy.sample()
            # Samples trajectories in parallel
            # For evaluation we can treat greedy and non-greedy the same: in both cases we just simulate
            # num_test_trajectories acquisition trajectories in parallel for each slice in the batch, and store
            # the average cross entropy score every time step.
            mask, masked_kspace, zf, image_input, outputs = compute_next_step_inference(infer_model, kspace,
                                                                               masked_kspace, mask, actions, args.use_feature_map)

            cross_entropy_scores, metrics_scores = compute_scores(args, outputs, label)
            # assert len(cross_entropy_scores) == 2
            # cross_entropy_scores = cross_entropy_scores.mean(-1).sum()
            accuracy_scores = metrics_scores['accuracy']


            # Append to lists
            batch_cross_entropy.append(cross_entropy_scores)
            batch_accuracy.append(accuracy_scores)



        # shape of al_steps
        cross_entropy += np.array(batch_cross_entropy)
        accuracy += np.array(batch_accuracy)
        # batch_confusion_matrix += np.array(batch_confusion_matrix) need to add more metrics

    cross_entropy /= tbs
    accuracy /= tbs

    # Logging
    if partition in ['Val', 'Train']:
        for step, val in enumerate(cross_entropy):
            writer.add_scalar(f'{partition}cross_entropy_step{step}', val, epoch)
            writer.add_scalar(f'{partition}accuracy_step{step}', accuracy[step], epoch)

        if args.wandb:
            wandb.log({f'{partition.lower()}_cross_entropy': {str(key): val for key, val in enumerate(cross_entropy)}}, step=epoch + 1)
            wandb.log({f'{partition.lower()}_accuracy': {str(key): val for key, val in enumerate(accuracy)}}, step=epoch + 1)

    elif partition == 'Test':
        # Only computed once, so loop over all epochs for wandb logging
        if args.wandb:
            for epoch in range(args.num_epochs):
                wandb.log({f'{partition.lower()}_cross_entropy': {str(key): val for key, val in enumerate(cross_entropy)}},
                    step=epoch + 1)
                wandb.log({f'{partition.lower()}_accuracy': {str(key): val for key, val in enumerate(accuracy)}},
                          step=epoch + 1)

    else:
        raise ValueError(f"'partition' should be in ['Train', 'Val', 'Test'], not: {partition}")

    return cross_entropy, accuracy, time.perf_counter() - start


def train_and_eval(args, infer_args, infer_model):
    """
    Wrapper for training and evaluation of policy model.

    :param args: Argument object, containing hyperparameters for training and evaluation.
    :param recon_args: reconstruction model arguments.
    :param recon_model: reconstruction model.
    """
    if args.resume:
        # Check that this works
        resumed = True
        new_run_dir = args.policy_model_checkpoint.parent
        data_path = args.data_path
        # In case models have been moved to a different machine, make sure the path to the recon model is the
        # path provided.
        infer_model_checkpoint = args.infer_model_checkpoint

        model, args, start_epoch, optimiser = load_policy_model(pathlib.Path(args.policy_model_checkpoint), optim=True)

        args.old_run_dir = args.run_dir
        args.old_infer_model_checkpoint = args.infer_model_checkpoint
        args.old_data_path = args.data_path

        args.infer_model_checkpoint = infer_model_checkpoint
        args.run_dir = new_run_dir
        args.data_path = data_path
        args.resume = True
    else:
        resumed = False
        # Improvement model to train
        model = build_policy_model(args)
        # Add mask parameters for training
        args = add_mask_params(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimiser = build_optim(args, model.parameters())
        start_epoch = 0
        # Create directory to store results in
        savestr = '{}_res{}_al{}_accel{}_k{}_{}_{}'.format(args.dataset, args.resolution, args.acquisition_steps,
                                                           args.accelerations, args.num_trajectories,
                                                           datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                                                           ''.join(choice(ascii_uppercase) for _ in range(5)))
        args.run_dir = args.exp_dir / savestr
        args.run_dir.mkdir(parents=True, exist_ok=False)

    args.resumed = resumed

    if args.wandb:
        allow_val_change = args.resumed  # only allow changes if resumed: otherwise something is wrong.
        wandb.config.update(args, allow_val_change=allow_val_change)
        wandb.watch(model, log='all')

    # Logging
    logging.info(infer_model)
    logging.info(model)
    # Save arguments for bookkeeping
    args_dict = {key: str(value) for key, value in args.__dict__.items()
                 if not key.startswith('__') and not callable(key)}
    save_json(args.run_dir / 'args.json', args_dict)

    # Initialise summary writer
    writer = SummaryWriter(log_dir=args.run_dir / 'summary')

    # Parameter counting
    logging.info('Inference model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(infer_model), count_trainable_parameters(infer_model),
        count_untrainable_parameters(infer_model)))
    logging.info('Policy model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))

    if args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, args.lr_step_size, args.lr_gamma)
    elif args.scheduler_type == 'multistep':
        if not isinstance(args.lr_multi_step_size, list):
            args.lr_multi_step_size = [args.lr_multi_step_size]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, args.lr_multi_step_size, args.lr_gamma)
    else:
        raise ValueError("{} is not a valid scheduler choice ('step', 'multistep')".format(args.scheduler_type))

    # Create data loaders
    train_loader = create_data_loader(args, 'train', shuffle=True)
    dev_loader = create_data_loader(args, 'val', shuffle=False)

    if not args.resume:
        if args.do_train_ssim:
            do_and_log_evaluation(args, -1, infer_model, model, train_loader, writer, 'Train')
        do_and_log_evaluation(args, -1, infer_model, model, dev_loader, writer, 'Val')

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, train_time = train_epoch(args, epoch, infer_model, model, train_loader, optimiser, writer)
        logging.info(
            f'Epoch = [{epoch+1:3d}/{args.num_epochs:3d}] TrainLoss = {train_loss:.3g} TrainTime = {train_time:.2f}s '
        )

        if args.do_train_ssim:
            do_and_log_evaluation(args, epoch, infer_model, model, train_loader, writer, 'Train')
        do_and_log_evaluation(args, epoch, infer_model, model, dev_loader, writer, 'Val')

        scheduler.step()
        save_policy_model(args, args.run_dir, epoch, model, optimiser)
    writer.close()


def do_and_log_evaluation(args, epoch, infer_model, model, loader, writer, partition):
    """
    Helper function for logging.
    """
    cross_entropy, accuracy, score_time = evaluate(args, epoch, infer_model, model, loader, writer, partition)
    cross_entropy_str = ", ".join(["{}: {:.4f}".format(i, l) for i, l in enumerate(cross_entropy)])
    accuracy_str = ", ".join(["{}: {:.3f}".format(i, l) for i, l in enumerate(accuracy)])
    logging.info(f'{partition}Cross Entropy = [{cross_entropy_str}]')
    logging.info(f'{partition}Accuracy = [{accuracy_str}]')
    logging.info(f'{partition}ScoreTime = {score_time:.2f}s')


def test(args, infer_model):
    """
    Performs evaluation of a pre-trained policy model.

    :param args: Argument object containing evaluation parameters.
    :param recon_model: reconstruction model.
    """
    model, policy_args = load_policy_model(pathlib.Path(args.policy_model_checkpoint))

    # Overwrite number of trajectories to test on
    policy_args.num_test_trajectories = args.num_test_trajectories
    if args.data_path is not None:  # Overwrite data path if provided
        policy_args.data_path = args.data_path

    # Logging of policy model
    logging.info(args)
    logging.info(infer_model)
    logging.info(model)
    if args.wandb:
        wandb.config.update(args)
        wandb.watch(model, log='all')
    # Initialise summary writer
    writer = SummaryWriter(log_dir=policy_args.run_dir / 'summary')

    # Parameter counting
    logging.info('Inference model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(infer_model), count_trainable_parameters(infer_model),
        count_untrainable_parameters(infer_model)))
    logging.info('Policy model parameters: total {}, of which {} trainable and {} untrainable'.format(
        count_parameters(model), count_trainable_parameters(model), count_untrainable_parameters(model)))

    # Create data loader
    test_loader = create_data_loader(policy_args, 'test', shuffle=False)
    # test_data_range_dict = create_data_range_dict(policy_args, test_loader)

    do_and_log_evaluation(policy_args, -1, infer_model, model, test_loader, writer, 'Test')

    writer.close()


def main(args):
    """
    Wrapper for training and testing of policy models.
    """
    logging.info(args)
    # Reconstruction model
    infer_args, infer_model = load_infer_model(args)
    infer_model = infer_model.to(args.device)
    ##load classificatio model instead and save classification parameters

    # Policy model to train
    if args.do_train:
        train_and_eval(args, infer_args, infer_model)
    else:
        test(args, infer_model)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=pathlib.Path, default='./Dataset/knee')
    parser.add_argument('--center_fractions', type=float, nargs='+', default=[0.1])
    parser.add_argument('--accelerations', type=int, nargs='+', default=[8])
    parser.add_argument('--resolution', type=list, default=356)
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--infer_model_checkpoint', type=str, default='../classification/log/0123_modif_res50_center01_multi_MT_knee_dropout/checkpoints/epoch=29-step=49770.ckpt')
    parser.add_argument('--use_feature_map', type=bool, default=False)
    parser.add_argument('--use_grad_campp', type=bool, default=False)
    parser.add_argument('--feature_map_layer', type=str, default='layer4')
    parser.add_argument('--dataset', default='knee', help='Dataset type to use.')

    parser.add_argument('--acquisition', type=str2none, default=None,
                        help='Use only volumes acquired using the provided acquisition method. Options are: '
                             'CORPD_FBK, CORPDFS_FBK (fat-suppressed), and not provided (both used).')
    parser.add_argument('--num_trajectories', type=int, default= 8, help='Number of actions to sample every acquisition '
                        'step during training.')
    parser.add_argument('--report_interval', type=int, default=1000, help='Period of loss reporting')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp_dir', type=pathlib.Path, default='./log/0130_zf_MT_center01',
                        help='Directory where model and results should be saved. Will create a timestamped folder '
                        'in provided directory each run')
    parser.add_argument('--reciprocals_in_center', nargs='+', default=[1], type=float,
                        help='Inverse fraction of rows (after subsampling) that should be in the center. E.g. if half '
                             'of the sampled rows should be in the center, this should be set to 2. All combinations '
                             'of acceleration and reciprocals-in-center will be used during training (every epoch a '
                             'volume randomly gets assigned an acceleration and center fraction.')
    parser.add_argument('--acquisition_steps', default=16, type=int, help='Acquisition steps to train for per image.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of ConvNet layers. Note that setting '
                        'this too high will cause size mismatch errors, due to even-odd errors in calculation for '
                        'layer size post-flattening (due to max pooling).')
    parser.add_argument('--drop_prob', type=float, default=0.2, help='Dropout probability')

    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Strength of weight decay regularization.')
    parser.add_argument('--center_volume', type=str2bool, default=True,
                        help='If set, only the center slices of a volume will be included in the dataset. This '
                             'removes the most noisy images from the data.')
    parser.add_argument('--data_parallel', type=str2bool, default=True,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--do_train_ssim', type=str2bool, default=False,
                        help='Whether to compute SSIM values on training data.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--seed', default=0, type=int, help='Seed for random number generators. '
                                                            'Set to 0 to use random seed.')
    parser.add_argument('--num_chans', type=int, default=16, help='Number of ConvNet channels in first layer.')
    parser.add_argument('--fc_size', default=256, type=int, help='Size (width) of fully connected layer(s).')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--scheduler_type', type=str, choices=['step', 'multistep'], default='step',
                        help='Number of training epochs')
    parser.add_argument('--lr_step_size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr_multi_step_size', nargs='+', type=int, default=[10, 20, 30, 40],
                        help='Epoch at which to decay the lr if using multistep scheduler.')
    parser.add_argument('--model_type', type=str, default='greedy', choices=['greedy', 'nongreedy'],
                        help="'greedy' to train greedy model, 'nongreedy' to train non-greedy model")
    parser.add_argument('--batches_step', type=int, default=1,
                        help='Number of dataloader batches to compute before doing an optimiser step. This is mostly '
                             'used to train non-greedy models with larger batch sizes.')
    parser.add_argument('--no_baseline', type=str2bool, default=False,
                        help="Whether to not use a reward baseline at all. Currently only implemented for 'greedy'.")
    parser.add_argument('--gamma', type=float, default=1,
                        help='Discount factor in RL. Currently only used for non-greedy training.')
    parser.add_argument('--milestones', nargs='+', type=int, default=[0, 9, 19, 29, 39, 49],
                        help='Epochs at which to save model separately.')

    parser.add_argument('--do_train', type=str2bool, default=True,
                        help='Whether to do training or testing.')
    parser.add_argument('--policy_model_checkpoint', type=pathlib.Path, default=None,
                        help='Path to a pretrained policy model if do_train is False (testing).')

    parser.add_argument('--wandb',  type=str2bool, default=False,
                        help='Whether to use wandb logging for this run.')
    parser.add_argument('--project',  type=str2none, default=None,
                        help='Wandb project name to use.')

    parser.add_argument('--resume',  type=str2bool, default=False,
                        help='Continue training previous run?')
    parser.add_argument('--run_id', type=str2none, default=None,
                        help='Wandb run_id to continue training from.')

    parser.add_argument('--num_test_trajectories', type=int, default=1,
                        help='Number of trajectories to use when testing sampling policy.')
    parser.add_argument('--test_multi',  type=str2bool, default=False,
                        help='Test multiple models in one script')
    parser.add_argument('--policy_model_list', nargs='+', type=str, default=[None],
                        help='List of policy model paths for multi-testing.')

    return parser


def wrap_main(args):
    """
    Wrapper for the entire script. Performs some setup, such as setting seed and starting wandb.
    """
    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)

    args.milestones = args.milestones + [0, args.num_epochs - 1]

    if args.wandb:
        if args.resume:
            assert args.run_id is not None, "run_id must be given if resuming with wandb."
            wandb.init(project=args.project, resume=args.run_id)
        elif args.test_multi:
            wandb.init(project=args.project, reinit=True)
        else:
            wandb.init(project=args.project, config=args)

    # To get reproducible behaviour, additionally set args.num_workers = 0 and disable cudnn
    # torch.backends.cudnn.enabled = False

    main(args)


if __name__ == '__main__':
    # To fix known issue with h5py + multiprocessing
    # See: https://discuss.pytorch.org/t/incorrect-data-using-h5py-with-dataloader/7079/2?u=ptrblck
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')

    base_args = create_arg_parser().parse_args()

    # Testing multiple policy models with one script
    if base_args.test_multi:
        assert not base_args.do_train, "Doing multiple model testing: do_train must be False."
        assert base_args.policy_model_list[0] is not None, ("Doing multiple model testing: must "
                                                            "have list of policy models.")

        for model in base_args.policy_model_list:
            args = copy.deepcopy(base_args)
            args.policy_model_checkpoint = model
            wrap_main(args)
            wandb.join()

    else:
        wrap_main(base_args)
