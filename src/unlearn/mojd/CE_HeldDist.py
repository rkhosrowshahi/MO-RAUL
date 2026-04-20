import sys
import time
import itertools
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchjd.aggregation import PCGrad, UPGrad, AlignedMTL, DualProj, IMTLG, MGDA, Mean, CAGrad, NashMTL
import utils

from ..impl import iterative_unlearn
from .utils import assert_split_forward_if_mtl, jd_backward

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def forget_kld_loss(logits_prior, labels_prior, logits_posterior, labels_posterior, epsilon=1e-10):
    """
    Compute class-wise KL divergence loss between prior (forget) and posterior (validation) distributions.
    For each class, computes KL divergence between average predictions on forget samples vs validation samples,
    then returns the mean KL divergence across all classes.
    
    Args:
        logits_prior (torch.Tensor): Model logits on forget data, shape (batch_size_prior, num_classes).
        labels_prior (torch.Tensor): True labels for forget data, shape (batch_size_prior,).
        logits_posterior (torch.Tensor): Model logits on validation data, shape (batch_size_posterior, num_classes).
        labels_posterior (torch.Tensor): True labels for validation data, shape (batch_size_posterior,).
        epsilon (float): Small value to clamp probabilities, default 1e-10.
    
    Returns:
        torch.Tensor: Mean KL divergence loss across classes.
    """
    num_classes = logits_posterior.size(1)
    
    # Convert logits to probabilities
    probs_prior = torch.softmax(logits_prior, dim=1)  # Shape: (batch_size_prior, num_classes)
    probs_posterior = torch.softmax(logits_posterior, dim=1)  # Shape: (batch_size_posterior, num_classes)
    
    kl_losses = []
    
    # Compute KL divergence for each class
    for class_idx in range(num_classes):
        # Find samples belonging to this class in prior (forget data)
        prior_mask = (labels_prior == class_idx)
        # Find samples belonging to this class in posterior (validation data)
        posterior_mask = (labels_posterior == class_idx)
        
        # Skip if class not present in either dataset
        if not prior_mask.any() or not posterior_mask.any():
            continue
        
        # Get average probability distribution for this class
        # Average over all forget samples of this class
        prior_class_probs = probs_prior[prior_mask].mean(dim=0)  # Shape: (num_classes,)
        # Average over all validation samples of this class
        posterior_class_probs = probs_posterior[posterior_mask].mean(dim=0)  # Shape: (num_classes,)
        
        # Clamp probabilities for numerical stability
        prior_class_probs = prior_class_probs.clamp(min=epsilon, max=1-epsilon)
        posterior_class_probs = posterior_class_probs.clamp(min=epsilon, max=1-epsilon)
        
        # Compute KL divergence: KL(prior || posterior) for this class
        # KL(P || Q) = sum(P * log(P/Q))
        kl_div = (prior_class_probs * (torch.log(prior_class_probs) - torch.log(posterior_class_probs))).sum()
        kl_losses.append(kl_div)
    
    # Return mean KL divergence across all classes
    if len(kl_losses) > 0:
        return torch.stack(kl_losses).mean()
    else:
        return torch.tensor(0.0, device=logits_prior.device)


def MO_JD_iter(
    data_loaders, model, criterion, optimizer, epoch, args, total_steps=0, aggregator=None, mask=None, wandb_run=None
):
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    val_loader = data_loaders["val"]

    loss_retain = utils.AverageMeter()
    top1_retain = utils.AverageMeter()
    loss_forget = utils.AverageMeter()
    top1_forget = utils.AverageMeter()
    loss_kl = utils.AverageMeter()
    loss_total = utils.AverageMeter()
    top1_total = utils.AverageMeter()

    ce_criterion = nn.CrossEntropyLoss()

    # switch to train mode
    model.train()

    agg = None
    if aggregator == "mean":
        agg = Mean()
    elif aggregator == "pcgrad":
        agg = PCGrad()
    elif aggregator == "upgrad":
        agg = UPGrad()
    elif aggregator == "amtl":
        agg = AlignedMTL()
    elif aggregator == "cagrad":
        agg = CAGrad()
    elif aggregator == "dualproj":
        agg = DualProj()
    elif aggregator == "imtl":
        agg = IMTLG()
    elif aggregator == "mgda":
        agg = MGDA()
    elif aggregator == "nashmtl":
        agg = NashMTL()
    else:
        raise NotImplementedError(f"Aggregator {aggregator} not implemented!")

    bm = getattr(args, "backward", "full")
    assert_split_forward_if_mtl(model, bm)

    retain_loader_iterator = itertools.cycle(retain_loader)
    forget_loader_iterator = itertools.cycle(forget_loader)
    val_loader_iterator = itertools.cycle(val_loader)
    
    i = 0
    num_batches = len(retain_loader)
    if num_batches < args.print_freq:
        args.print_freq = num_batches // 2

    start = time.time()
    if args.imagenet_arch:
        for i in range(num_batches):
            retain_image, retain_target = next(retain_loader_iterator)
            forget_image, forget_target = next(forget_loader_iterator)
            val_image, val_target = next(val_loader_iterator)
            retain_image, retain_target = retain_image.cuda(), retain_target.cuda()
            forget_image, forget_target = forget_image.cuda(), forget_target.cuda()
            val_image, val_target = val_image.cuda(), val_target.cuda()

            # Objective 1: Retain performance - minimize CE on retain data
            # Objective 2: Forget via KL - make forget samples match validation distribution
            if bm == "mtl":
                features_r = model.forward_features(retain_image)
                logits_r = model.forward_classifier(features_r)
                features_f = model.forward_features(forget_image)
                logits_f = model.forward_classifier(features_f)
                features_v = model.forward_features(val_image)
                logits_v = model.forward_classifier(features_v)
            else:
                logits_r = model(retain_image)
                logits_f = model(forget_image)
                logits_v = model(val_image)
            ce_loss_r = ce_criterion(logits_r, retain_target)
            kl_loss_f = forget_kld_loss(logits_f, forget_target, logits_v, val_target)

            losses = [ce_loss_r, kl_loss_f]

            optimizer.zero_grad()
            jd_backward(
                losses,
                model,
                agg,
                bm,
                features=[features_r, features_f, features_v] if bm == "mtl" else None,
            )
            optimizer.step()

            # measure accuracy and record loss
            prec1_retain = utils.accuracy(logits_r.data, retain_target)[0]

            loss_retain.update(ce_loss_r.item(), retain_image.size(0))
            top1_retain.update(prec1_retain.item(), retain_image.size(0))

             # measure accuracy and record loss for forget
            prec1_forget = utils.accuracy(logits_f.data, forget_target)[0]

            loss_kl.update(kl_loss_f.item(), forget_image.size(0))
            top1_forget.update(prec1_forget.item(), forget_image.size(0))

            sum_loss = ce_loss_r + kl_loss_f
            loss_total.update(sum_loss.item(), retain_image.size(0) + forget_image.size(0))
            
            # Log step-level metrics to wandb (training phase - increment after)
            if wandb_run is not None:
                wandb_run.log({
                    "train/retain_ce_current": loss_retain.val,
                    "train/retain_ce_running": loss_retain.avg,
                    "train/retain_accuracy_current": top1_retain.val,
                    "train/retain_accuracy_running": top1_retain.avg,
                    "train/forget_kl_current": loss_kl.val,
                    "train/forget_kl_running": loss_kl.avg,
                    "train/forget_accuracy_current": top1_forget.val,
                    "train/forget_accuracy_running": top1_forget.avg,
                    "train/total_loss_current": loss_total.val,
                    "train/total_loss_running": loss_total.avg,
                }, step=total_steps)
            total_steps += 1

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    f"Epoch: [{epoch}][{i}/{num_batches}]\t"
                    f"Retain CE {loss_retain.val:.4f} ({loss_retain.avg:.4f})\t"
                    f"Retain Accuracy {top1_retain.val:.3f} ({top1_retain.avg:.3f})\t"
                    f"Forget KL {loss_kl.val:.4f} ({loss_kl.avg:.4f})\t"
                    f"Forget Accuracy {top1_forget.val:.3f} ({top1_forget.avg:.3f})\t"
                    f"Total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t"
                    f"Time {end - start:.2f}"
                )
                start = time.time()
    else:
        for i in range(num_batches):
            retain_image, retain_target = next(retain_loader_iterator)
            forget_image, forget_target = next(forget_loader_iterator)
            val_image, val_target = next(val_loader_iterator)
            retain_image, retain_target = retain_image.cuda(), retain_target.cuda()
            forget_image, forget_target = forget_image.cuda(), forget_target.cuda()
            val_image, val_target = val_image.cuda(), val_target.cuda()

            # Objective 1: Retain performance - minimize CE on retain data
            # Objective 2: Forget via KL - make forget samples match validation distribution
            if bm == "mtl":
                features_r = model.forward_features(retain_image)
                logits_r = model.forward_classifier(features_r)
                features_f = model.forward_features(forget_image)
                logits_f = model.forward_classifier(features_f)
                features_v = model.forward_features(val_image)
                logits_v = model.forward_classifier(features_v)
            else:
                logits_r = model(retain_image)
                logits_f = model(forget_image)
                logits_v = model(val_image)
            ce_loss_retain = ce_criterion(logits_r, retain_target)
            kl_loss_forget = forget_kld_loss(logits_f, forget_target, logits_v, val_target)
            
            losses = [ce_loss_retain, kl_loss_forget]

            optimizer.zero_grad()
            jd_backward(
                losses,
                model,
                agg,
                bm,
                features=[features_r, features_f, features_v] if bm == "mtl" else None,
            )
            optimizer.step()

            # measure accuracy and record loss
            prec1_retain = utils.accuracy(logits_r.data, retain_target)[0]

            loss_retain.update(ce_loss_retain.item(), retain_image.size(0))
            top1_retain.update(prec1_retain.item(), retain_image.size(0))

             # measure accuracy and record loss for forget
            prec1_forget = utils.accuracy(logits_f.data, forget_target)[0]

            loss_kl.update(kl_loss_forget.item(), forget_image.size(0))
            top1_forget.update(prec1_forget.item(), forget_image.size(0))

            sum_loss = ce_loss_retain + kl_loss_forget
            loss_total.update(sum_loss.item(), retain_image.size(0) + forget_image.size(0))
            
            # Log step-level metrics to wandb (training phase - increment after)
            if wandb_run is not None:
                wandb_run.log({
                    "train/retain_ce_current": loss_retain.val,
                    "train/retain_ce_running": loss_retain.avg,
                    "train/retain_accuracy_current": top1_retain.val,
                    "train/retain_accuracy_running": top1_retain.avg,
                    "train/forget_kl_current": loss_kl.val,
                    "train/forget_kl_running": loss_kl.avg,
                    "train/forget_accuracy_current": top1_forget.val,
                    "train/forget_accuracy_running": top1_forget.avg,
                    "train/total_loss_current": loss_total.val,
                    "train/total_loss_running": loss_total.avg,
                }, step=total_steps)
            total_steps += 1

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    f"Epoch: [{epoch}][{i}/{num_batches}]\t"
                    f"Retain CE {loss_retain.val:.4f} ({loss_retain.avg:.4f})\t"
                    f"Retain Accuracy {top1_retain.val:.3f} ({top1_retain.avg:.3f})\t"
                    f"Forget KL {loss_kl.val:.4f} ({loss_kl.avg:.4f})\t"
                    f"Forget Accuracy {top1_forget.val:.3f} ({top1_forget.avg:.3f})\t"
                    f"Total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t"
                    f"Time {end - start:.2f}"
                )
                start = time.time()

    return total_steps, top1_retain.avg


@iterative_unlearn
def MO_JD_CE_HeldDist(data_loaders, model, criterion, optimizer, epoch, args, total_steps=0, mask=None, wandb_run=None):
    return MO_JD_iter(
        data_loaders,
        model,
        criterion,
        optimizer,
        epoch,
        args,
        total_steps=total_steps,
        aggregator=args.mo_aggregator,
        mask=mask,
        wandb_run=wandb_run,
    )
