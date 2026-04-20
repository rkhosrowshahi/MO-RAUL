import torch
import time
from copy import deepcopy
import utils
from .impl import iterative_unlearn
import numpy as np

@iterative_unlearn
def RL_proximal(data_loaders, model, criterion, optimizer, epoch, args, total_steps=0, mask=None, wandb_run=None):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)
    mask_ratio = args.mask_ratio
    
    # concat all params
    init_params = torch.concat([param.view(-1) for param in model.parameters()], dim=0)
    n_params = init_params.numel()        
    total_steps_for_proximal = args.unlearn_epochs * (len(forget_loader) + len(retain_loader))
    
    if args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "tiny_imagenet":
        forget_dataset.targets = np.random.randint(0, args.num_classes, forget_dataset.targets.shape)
    
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
            # compute output
            output_clean = model(image)
            loss = criterion(output_clean, target)
      
            optimizer.zero_grad()
            loss.backward()

            
            optimizer.step()
                  
            ratio = int(mask_ratio * ((total_steps_for_proximal - (epoch * (len(forget_loader) + len(retain_loader)) + 1)) / total_steps_for_proximal * n_params))           
            params = torch.concat([param.view(-1) for param in model.parameters()], dim=0)
            diff_params = params - init_params
            threshold = -torch.topk(-diff_params.abs(), ratio)[0][-1]
            params = torch.where(diff_params > threshold, params - threshold, 
                                        torch.where(diff_params < -threshold, params + threshold, init_params))
            # update params
            for name, param in model.named_parameters():
                param.data = params[:param.numel()].view(param.shape)
                params = params[param.numel():]
      
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
                    f"Steps: [{epoch}]\t"
                    f"Epoch: [{epoch}][{i}/{loader_len}]\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    f"Time {end - start:.2f}"
                )
                start = time.time()
      
    elif args.dataset == "svhn":
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
            optimizer.step()
            
            ratio = int(mask_ratio * ((total_steps_for_proximal - (epoch * (len(forget_loader) + len(retain_loader)) + 1)) / total_steps_for_proximal * n_params))           
            params = torch.concat([param.view(-1) for param in model.parameters()], dim=0)
            diff_params = params - init_params
            threshold = -torch.topk(-diff_params.abs(), ratio)[0][-1]
            params = torch.where(diff_params > threshold, params - threshold, 
                                        torch.where(diff_params < -threshold, params + threshold, init_params))
            # update params
            for name, param in model.named_parameters():
                param.data = params[:param.numel()].view(param.shape)
                params = params[param.numel():]
            
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
            optimizer.step()
            
            ratio = int(mask_ratio * ((total_steps_for_proximal - (epoch * (len(forget_loader) + len(retain_loader)) + i + 1)) / total_steps_for_proximal * n_params))           
            params = torch.concat([param.view(-1) for param in model.parameters()], dim=0)
            diff_params = params - init_params
            threshold = -torch.topk(-diff_params.abs(), ratio)[0][-1]
            params = torch.where(diff_params > threshold, params - threshold, 
                                        torch.where(diff_params < -threshold, params + threshold, init_params))
            # update params
            for name, param in model.named_parameters():
                param.data = params[:param.numel()].view(param.shape)
                params = params[param.numel():]
            
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
                   f"Steps: [{epoch}]\t"
                   f"Epoch: [{epoch}][{i}/{loader_len}]\t"
                   f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                   f"Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                   f"Time {end - start:.2f}"
               )
               start = time.time()
               
    return total_steps, top1.avg