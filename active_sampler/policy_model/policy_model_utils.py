import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
from .policy_model_def import build_policy_model
sys.path.append('..')
from utils.utils import build_optim
from utils.complex import complex_abs
from utils.fft import fft2c, ifft2c
from utils.transform_utils import to_tensor, complex_center_crop, normalize, normalize_instance
from utils.torch_metrics import compute_cross_entropy,compute_batch_metrics

def save_policy_model(args, exp_dir, epoch, model, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    if epoch in args.milestones:
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'exp_dir': exp_dir
            },
            f=exp_dir / f'model_{epoch}.pt'
        )


def load_policy_model(checkpoint_file, optim=False):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_policy_model(args)

    if not optim:
        # No gradients for this model
        for param in model.parameters():
            param.requires_grad = False

    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    start_epoch = checkpoint['epoch']

    if optim:
        optimizer = build_optim(args, model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        return model, args, start_epoch, optimizer

    del checkpoint
    return model, args


def get_new_zf(masked_kspace_batch):
    # Inverse Fourier Transform to get zero filled solution
    image_batch = complex_abs(ifft2c(masked_kspace_batch))
    # Normalize input
    image_batch, means, stds = normalize_instance(image_batch, eps=1e-11)
    image_batch = image_batch.clamp(-6, 6)
    image_batch = image_batch.expand(-1, 3, -1, -1)
    return image_batch, means, stds


def acquire_rows_in_batch_parallel(k, mk, mask, to_acquire):
    if mask.size(1) == mk.size(1) == to_acquire.size(1):
        # print(to_acquire.size())
        # Two cases:
        # 1) We are only requesting a single k-space column to acquire per batch.
        # 2) We are requesting multiple k-space columns per batch, and we are already in a trajectory of the non-greedy
        # model: every column in to_acquire corresponds to an existing trajectory that we have sampled the next
        # column for.
        m_exp = mask
        mk_exp = mk
    else:
        # We have to initialise trajectories: every row in to_acquire corresponds to a trajectory.
        m_exp = mask.repeat(1, to_acquire.size(1), 1, 1, 1)
        mk_exp = mk.repeat(1, to_acquire.size(1), 1, 1, 1)
    # Loop over slices in batch
    for sl, rows in enumerate(to_acquire):
        # Loop over indices to acquire
        for index, row in enumerate(rows):  # Will only be single index if first case (see comment above)
            m_exp[sl, index, :, row.item(), :] = 1.
            mk_exp[sl, index, :, row.item(), :] = k[sl, 0, :, row.item(), :]
    return m_exp, mk_exp


def compute_next_step_inference(infer_model, kspace, masked_kspace, mask, next_rows,use_feature_map):
    # This computation is done by reshaping the masked k-space tensor to (batch . num_trajectories x 1 x 640 x res)
    # and then reshaping back after performing a reconstruction.
    image_input = []
    outputs = []
    zf = []
    mask, masked_kspace = acquire_rows_in_batch_parallel(kspace, masked_kspace, mask, next_rows)
    channel_size = masked_kspace.shape[1]
    res = masked_kspace.size(-2)
    # Combine batch and channel dimension for parallel computation if necessary
    masked_kspace = masked_kspace.view(mask.size(0), channel_size, 640, res, 2)
    for num in range(channel_size):
        single_mp = masked_kspace[:,num].unsqueeze(1)
        single_zf, _, _ = get_new_zf(single_mp)
    # Base inference model forward pass
        if use_feature_map:
            single_feature_map, single_outputs = infer_model(single_zf)
        else:
            single_outputs = infer_model(single_zf)
            single_feature_map = single_zf[:, 0, :, :].unsqueeze(1)
        image_input.append(single_feature_map)
        outputs.append(single_outputs)
        zf.append(single_zf[:, 0, :, :].unsqueeze(1))
    # Reshape back to B X C (=parallel acquisitions) x H x W
    outputs = torch.stack(outputs, dim=1)
    image_input = torch.stack(image_input, dim=1)
    zf = torch.stack(zf, dim=1)

    image_input = image_input.view(mask.size(0), channel_size, 640, res)
    zf = zf.view(mask.size(0),  channel_size, 640, res)
    masked_kspace = masked_kspace.view(mask.size(0), channel_size, 640, res, 2)
    return mask, masked_kspace, zf, image_input, outputs.squeeze()


def get_policy_probs(model, input_image, mask):
    # mask size[32, 1, 356, 1] torch.Size([16, 1, 1, 128, 1])
    channel_size = mask.shape[1]
    res = mask.size(-2)
    # Reshape trajectory dimension into batch dimension for parallel forward pass
    # print(np.shape(input_image), np.shape(mask))
    input_image = input_image.view(mask.size(0) * channel_size, 1, 640, res)
    # Obtain policy model logits
    output = model(input_image)
    # Reshape trajectories back into their own dimension
    output = output.view(mask.size(0), channel_size, res)  #[batch,1,res]
    # Mask already acquired rows by setting logits to very negative numbers
    loss_mask = (mask == 0).squeeze(-1).squeeze(-2).float()
    logits = torch.where(loss_mask.byte(), output, -1e7 * torch.ones_like(output))
    # Softmax over 'logits' representing row scores
    probs = torch.nn.functional.softmax(logits - logits.max(dim=-1, keepdim=True)[0], dim=-1)
    # Also need this for sampling the next row at the end of this loop
    policy = torch.distributions.Categorical(probs)
    return policy, probs


def compute_scores(args, outputs, label):
    cross_entropy = compute_cross_entropy(outputs, label)
    metrics = compute_batch_metrics(outputs, label)
    return cross_entropy, metrics




def compute_backprop_trajectory(args, kspace, masked_kspace, mask, outputs, image_input, label, model, infer_model, step, action_list, logprob_list, reward_list):
    cross_entropy_scores = []
    criteria = nn.BCELoss(reduction='none')
    # Base score from which to calculate acquisition rewards
    base_scores = criteria(outputs, F.one_hot(label, outputs.shape[-1]).float()).mean(dim=-1)
    # Get policy and probabilities.
    policy, probs = get_policy_probs(model, image_input, mask)
    # Sample actions from the policy. For greedy (or at step = 0) we sample num_trajectories actions from the
    # current policy. For non-greedy with step > 0, we sample a single action for every of the num_trajectories
    # policies.
    # probs shape = batch x num_traj x res
    # actions shape = batch x num_traj
    # action_logprobs shape = batch x num_traj
    if step == 0 or args.model_type == 'greedy':  # probs has shape batch x 1 x res
        actions = torch.multinomial(probs.squeeze(1), args.num_trajectories, replacement=True)
        actions = actions.unsqueeze(1)  # batch x num_traj -> batch x 1 x num_traj
        # probs shape = batch x 1 x res
        action_logprobs = torch.log(torch.gather(probs, -1, actions)).squeeze(1)
        actions = actions.squeeze(1)
    else:  # Non-greedy model and step > 0: this means probs has shape batch x num_traj x res
        actions = policy.sample()
        actions = actions.unsqueeze(-1)  # batch x num_traj -> batch x num_traj x 1
        # probs shape = batch x num_traj x res
        action_logprobs = torch.log(torch.gather(probs, -1, actions)).squeeze(-1)
        actions = actions.squeeze(1)

    # Obtain rewards in parallel by taking actions in parallel
    mask, masked_kspace, zf, image_input, outputs = compute_next_step_inference(infer_model, kspace,
                                                                                masked_kspace, mask, actions,
                                                                                args.use_feature_map)
    for ch in range(masked_kspace.shape[1]):
        single_cross_entropy_scores = criteria(outputs[:, ch, :], F.one_hot(label, outputs.shape[-1]).float())
        cross_entropy_scores.append(single_cross_entropy_scores.mean(dim=-1))
    # batch x num_trajectories
    action_rewards = base_scores.unsqueeze(-1) - torch.stack(cross_entropy_scores,dim=0).transpose(1,0)
    # print(action_rewards.shape)
    # batch x 1
    avg_reward = torch.mean(action_rewards, dim=-1, keepdim=True)
    # Store for non-greedy model (we need the full return before we can do a backprop step)
    action_list.append(actions)
    logprob_list.append(action_logprobs)
    reward_list.append(action_rewards)

    if args.model_type == 'greedy':
        # batch x k
        if args.no_baseline:
            # No-baseline
            loss = -1 * (action_logprobs * action_rewards) / actions.size(-1)
        else:
            # Local baseline
            loss = -1 * (action_logprobs * (action_rewards - avg_reward)) / (actions.size(-1) - 1)
        # batch
        loss = loss.sum(dim=1)
        # Average over batch
        # Divide by batches_step to mimic taking mean over larger batch
        loss = loss.mean() / args.batches_step  # For consistency: we generally set batches_step to 1 for greedy
        loss.backward()

        # For greedy: initialise next step by randomly picking one of the measurements for every slice
        # For non-greedy we will continue with the parallel sampled rows stored in masked_kspace, and
        # with mask, zf, and recons.
        idx = random.randint(0, mask.shape[1] - 1)
        mask = mask[:, idx, :, :, :].unsqueeze(1)
        masked_kspace = masked_kspace[:, idx, :, :, :].unsqueeze(1)
        image_input = image_input[:, idx, :, :].unsqueeze(1)
        outputs = outputs[:, idx, :]

    elif step != args.acquisition_steps - 1:  # Non-greedy but don't have full return yet.
        loss = torch.zeros(1)  # For logging
    else:  # Final step, can compute non-greedy return
        reward_tensor = torch.stack(reward_list)
        for step, logprobs in enumerate(logprob_list):
            # Discount factor
            gamma_vec = [args.gamma ** (t - step) for t in range(step, args.acquisition_steps)]
            gamma_ten = torch.tensor(gamma_vec).unsqueeze(-1).unsqueeze(-1).to(args.device)
            # step x batch x 1
            avg_rewards_tensor = torch.mean(reward_tensor, dim=2, keepdim=True)
            # Get number of trajectories for correct average
            num_traj = logprobs.size(-1)
            # REINFORCE with self-baselines
            # batch x k
            # TODO: can also store transitions (s, a, r, s') pairs and recompute log probs when
            #  doing gradients? Takes less memory, but more compute: can this be efficiently
            #  batched?
            loss = -1 * (logprobs * torch.sum(
                gamma_ten * (reward_tensor[step:, :, :] - avg_rewards_tensor[step:, :, :]),
                dim=0)) / (num_traj - 1)
            # batch
            loss = loss.sum(dim=1)
            # Average over batch
            # Divide by batches_step to mimic taking mean over larger batch
            loss = loss.mean() / args.batches_step
            loss.backward()  # Store gradients
    return loss, mask, masked_kspace, image_input, outputs
