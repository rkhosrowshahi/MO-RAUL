import time
from copy import deepcopy

import numpy as np
import torch
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

@iterative_unlearn
def RL(data_loaders, model, criterion, optimizer, epoch, args, total_steps=0, mask=None, wandb_run=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)

    theta0 = None
    if mask:
        with torch.no_grad():
            theta0 = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
                if name in mask
            }
    
    if args.dataset == "cifar100" or args.dataset == "tiny_imagenet":
        try:
            forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
        except:
            forget_dataset.dataset.targets = np.random.randint(0, args.num_classes, len(forget_dataset.dataset.targets))
        retain_dataset = retain_loader.dataset
        train_dataset = torch.utils.data.ConcatDataset([forget_dataset,retain_dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
      
        for it, (image, target) in enumerate(train_loader):
            i = it + len(forget_loader)
            image = image.cuda()
            target = target.cuda()
            output_clean = model(image)

            loss = criterion(output_clean, target)
      
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
                    "train/loss_current": losses.val,
                    "train/loss_running": losses.avg,
                    "train/accuracy_current": top1.val,
                    "train/accuracy_running": top1.avg,
                }, step=total_steps)
            
            total_steps += 1
      
            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    f"Steps: [{total_steps}]\t"
                    f"Epoch: [{epoch}][{i}/{loader_len}]\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    f"Time {end - start:.2f}"
                )
                start = time.time()
      
    elif args.dataset == "cifar10" or args.dataset == "svhn":
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
      
        # switch to train mode
        model.train()
      
        start = time.time()
        loader_len = len(forget_loader) + len(retain_loader)
      
        if epoch < args.warmup:
            utils.warmup_lr(epoch, i+1, optimizer,
                            one_epoch_step=loader_len, args=args)
        
        for i, (image, target) in enumerate(forget_loader):
            image = image.cuda()
            target = torch.randint(0, args.num_classes, target.shape).cuda()
            
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            optimizer.zero_grad()
            loss.backward()
            
            if mask:
                _apply_mask_to_grads(model, mask)
            
            optimizer.step()
            
            if mask:
                _restore_masked_params(model, mask, theta0, optimizer)

            # Log step-level metrics for forget data to wandb
            if wandb_run is not None:
                wandb_run.log({
                    "train/forget_step": i,
                }, step=total_steps)
            
            total_steps += 1
            
        for i, (image, target) in enumerate(retain_loader):
            image = image.cuda()
            target = target.cuda()
            
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
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
                    "train/retain_loss_current": losses.val,
                    "train/retain_loss_running": losses.avg,
                    "train/retain_accuracy_current": top1.val,
                    "train/retain_accuracy_running": top1.avg,
                }, step=total_steps)
            
            total_steps += 1
            
            if (i + 1) % args.print_freq == 0:
               end = time.time()
               print(
                   f"Steps: [{total_steps}]\t"
                   f"Epoch: [{epoch}][{i}/{loader_len}]\t"
                   f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                   f"Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                   f"Time {end - start:.2f}"
               )
               start = time.time()

    return total_steps, top1.avg