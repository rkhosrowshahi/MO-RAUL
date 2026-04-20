import copy
import os
import time

import torch
import utils
from imagenet import get_x_y_from_data_dict


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def get_optimizer_and_scheduler(model, args):
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = utils.get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    return optimizer, scheduler


def train(train_loader, model, criterion, optimizer, epoch, args, total_steps=0, mask=None, l1=False, wandb_run=None):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )
            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            # Log step-level metrics to wandb
            if wandb_run is not None:
                wandb_run.log({
                    "train/retain_ce_current": losses.val,
                    "train/retain_ce_running": losses.avg,
                    "train/retain_accuracy_current": top1.val,
                    "train/retain_accuracy_running": top1.avg,
                }, step=total_steps)
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

        print(f"train_accuracy {top1.avg:.3f}")
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            # Log step-level metrics to wandb
            if wandb_run is not None:
                wandb_run.log({
                    "train/retain_ce_current": losses.val,
                    "train/retain_ce_running": losses.avg,
                    "train/retain_accuracy_current": top1.val,
                    "train/retain_accuracy_running": top1.avg,
                }, step=total_steps)
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

        print(f"train_accuracy {top1.avg:.3f}")

    return total_steps, top1.avg


def train_with_rewind(model, optimizer, scheduler, train_loader, criterion, args):
    """Train the model and return the state dict for rewinding purposes."""
    for epoch in range(args.epochs):
        train(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step()
    
    # Return the state dict for rewinding
    return model.state_dict()
