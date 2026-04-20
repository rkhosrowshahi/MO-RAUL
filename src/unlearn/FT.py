import sys
import time

import torch
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict

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

            param.data.mul_(mask_tensor).add_(theta0[name].to(param.device) * inv_mask_tensor)

            state = optimizer.state.get(param, None)
            if state is not None and "momentum_buffer" in state:
                state["momentum_buffer"].mul_(mask_tensor)


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def FT_iter(
    data_loaders, model, criterion, optimizer, epoch, args, total_steps=0, mask=None, with_l1=False, wandb_run=None
):
    train_loader = data_loaders["retain"]

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
    if args.imagenet_arch:
        device = (
            torch.device(f"cuda:{args.gpu}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # compute output
            output_clean = model(image)
            if epoch < args.unlearn_epochs - args.no_l1_epochs:
                current_alpha = args.alpha * (
                    1 - epoch / (args.unlearn_epochs - args.no_l1_epochs)
                )
            else:
                current_alpha = 0
            loss = criterion(output_clean, target)
            if with_l1:
                loss = loss + current_alpha * l1_regularization(model)
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
                log_dict = {
                    "train/retain_loss_current": losses.val,
                    "train/retain_loss_running": losses.avg,
                    "train/retain_accuracy_current": top1.val,
                    "train/retain_accuracy_running": top1.avg,
                }
                if with_l1:
                    log_dict["train/l1_alpha"] = current_alpha
                wandb_run.log(log_dict, step=total_steps)
            
            total_steps += 1

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    f"Steps: [{total_steps}]\t"
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    f"Time {end - start:.2f}"
                )
                start = time.time()
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()
            if epoch < args.unlearn_epochs - args.no_l1_epochs:
                current_alpha = args.alpha * (
                    1 - epoch / (args.unlearn_epochs - args.no_l1_epochs)
                )
            else:
                current_alpha = 0
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            if with_l1:
                loss += current_alpha * l1_regularization(model)

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
                log_dict = {
                    "train/retain_loss_current": losses.val,
                    "train/retain_loss_running": losses.avg,
                    "train/retain_accuracy_current": top1.val,
                    "train/retain_accuracy_running": top1.avg,
                }
                if with_l1:
                    log_dict["train/l1_alpha"] = current_alpha
                wandb_run.log(log_dict, step=total_steps)
            
            total_steps += 1

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    f"Steps: [{total_steps}]\t"
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    f"Time {end - start:.2f}"
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return total_steps, top1.avg


@iterative_unlearn
def FT(data_loaders, model, criterion, optimizer=None, epoch=None, args=None, total_steps=0, mask=None, wandb_run=None):
    return FT_iter(data_loaders, model, criterion, optimizer, epoch, args, total_steps=total_steps, mask=mask, wandb_run=wandb_run)


@iterative_unlearn
def FT_l1(data_loaders, model, criterion, optimizer=None, epoch=None, args=None, total_steps=0, mask=None, wandb_run=None):
    return FT_iter(
        data_loaders, model, criterion, optimizer, epoch, args, total_steps=total_steps, mask=mask, with_l1=True, wandb_run=wandb_run
    )
