import time

import torch
import torch.nn as nn
import utils

from .impl import iterative_unlearn

def _apply_mask_to_grads(model, mask):
    for name, param in model.named_parameters():
        if param.grad is not None:
            param.grad *= mask[name]


def _restore_masked_params(model, mask, theta0, optimizer):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in mask:
                continue

            mask_tensor = mask[name].to(device=param.device, dtype=param.dtype)
            inv_mask_tensor = 1 - mask_tensor
            if torch.count_nonzero(inv_mask_tensor) == 0:
                continue

            # Keep masked-out weights exactly at initialization value (theta0).
            param.data.mul_(mask_tensor).add_(theta0[name].to(param.device) * inv_mask_tensor)

            # Prevent momentum from reintroducing updates on masked-out coordinates.
            state = optimizer.state.get(param, None)
            if state is not None and "momentum_buffer" in state:
                state["momentum_buffer"].mul_(mask_tensor)


def expand_model(model):
    last_fc_name = None
    last_fc_layer = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_fc_name = name
            last_fc_layer = module

    if last_fc_name is None:
        raise ValueError("No Linear layer found in the model.")

    num_classes = last_fc_layer.out_features

    bias = last_fc_layer.bias is not None

    new_last_fc_layer = nn.Linear(
        in_features=last_fc_layer.in_features,
        out_features=num_classes + 1,
        bias=bias,
        device=last_fc_layer.weight.device,
        dtype=last_fc_layer.weight.dtype,
    )

    with torch.no_grad():
        new_last_fc_layer.weight[:-1] = last_fc_layer.weight
        if bias:
            new_last_fc_layer.bias[:-1] = last_fc_layer.bias

    parts = last_fc_name.split(".")
    current_module = model
    for part in parts[:-1]:
        current_module = getattr(current_module, part)
    setattr(current_module, parts[-1], new_last_fc_layer)


@iterative_unlearn
def boundary_expanding_iter(
    data_loaders, model, criterion, optimizer, epoch, args, total_steps=0, mask=None, wandb_run=None
):
    train_loader = data_loaders["forget"]

    theta0 = None
    if mask:
        with torch.no_grad():
            theta0 = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
                if name in mask
            }
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):
        image = image.cuda()
        target = target.cuda()

        target_label = torch.ones_like(target)
        target_label *= args.num_classes
        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target_label)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            _apply_mask_to_grads(model, mask)

        optimizer.step()

        if mask:
            _restore_masked_params(model, mask, theta0, optimizer)

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
        
        # Log step-level metrics to wandb
        if wandb_run is not None:
            wandb_run.log({
                "train/forget_loss_current": losses.val,
                "train/forget_loss_running": losses.avg,
                "train/forget_accuracy_current": top1.val,
                "train/forget_accuracy_running": top1.avg,
            }, step=total_steps)
        
        total_steps += 1

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                f"Steps: [{epoch}]\t"
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                f"Time {end - start:.2f}"
            )
            start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return total_steps, top1.avg


def boundary_expanding(data_loaders, model, criterion, args, mask=None, wandb_run=None):
    expand_model(model)
    return boundary_expanding_iter(data_loaders, model, criterion, args, mask=mask, wandb_run=wandb_run)
