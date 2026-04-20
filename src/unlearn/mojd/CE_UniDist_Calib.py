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


def kl_uniform_criterion(logits):
    """
    KL divergence between model predictions and uniform distribution.
    Pushes the model to output uniform predictions on forget set.
    """
    num_classes = logits.size(1)
    probs = torch.softmax(logits, dim=1)
    uniform_dist = torch.ones_like(probs) / num_classes
    # KL(P || Q) where P is model prediction, Q is uniform
    kl_div = torch.sum(probs * (torch.log(probs + 1e-10) - torch.log(uniform_dist)), dim=1)
    return torch.mean(kl_div)


def calibration_loss(logits, targets, smoothing=0.15):
    """
    Combines label smoothing with confidence penalty.
    Creates controlled uncertainty: majority correct, some errors.
    Used to reduce model confidence on forget set to create gradual unlearning.
    """
    # Label smoothing component
    num_classes = logits.size(1)
    confidence = 1.0 - smoothing
    
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), confidence)
    
    smooth_loss = torch.mean(torch.sum(-true_dist * log_probs, dim=1))
    
    # Confidence penalty component
    probs = F.softmax(logits, dim=1)
    max_probs = torch.max(probs, dim=1)[0]
    confidence_penalty = 0.1 * torch.mean(max_probs)
    
    return smooth_loss + confidence_penalty


def MO_JD_CE_UniDist_Calib_iter(
    data_loaders, model, criterion, optimizer, epoch, args, total_steps=0, aggregator=None, mask=None, wandb_run=None
):
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]

    loss_retain = utils.AverageMeter()
    top1_retain = utils.AverageMeter()
    loss_forget = utils.AverageMeter()
    top1_forget = utils.AverageMeter()
    loss_entropy_forget = utils.AverageMeter()
    top1_entropy_forget = utils.AverageMeter()
    loss_calibration = utils.AverageMeter()
    loss_total = utils.AverageMeter()
    top1_total = utils.AverageMeter()

    ce_criterion = nn.CrossEntropyLoss()

    # switch to train mode
    model.train()

    agg = None
    print(aggregator)
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

    
    i = 0
    num_batches = len(retain_loader)
    if num_batches < args.print_freq:
        args.print_freq = num_batches // 2

    start = time.time()
    if args.imagenet_arch:
        for i in range(num_batches):
            retain_image, retain_target = next(retain_loader_iterator)
            forget_image, forget_target = next(forget_loader_iterator)
            retain_image, retain_target = retain_image.cuda(), retain_target.cuda()
            forget_image, forget_target = forget_image.cuda(), forget_target.cuda()

            # compute output
            if bm == "mtl":
                features_r = model.forward_features(retain_image)
                logits_r = model.forward_classifier(features_r)
                features_f = model.forward_features(forget_image)
                logits_f = model.forward_classifier(features_f)
            else:
                logits_r = model(retain_image)
                logits_f = model(forget_image)
            ce_loss_r = ce_criterion(logits_r, retain_target)
            # KL divergence to push forget set predictions toward uniform distribution
            kl_uniform_forget = kl_uniform_criterion(logits_f)

            # Calibration loss to reduce confidence on forget set
            calib_loss = calibration_loss(logits_f, forget_target)

            losses = [ce_loss_r, kl_uniform_forget, calib_loss]

            optimizer.zero_grad()
            jd_backward(
                losses,
                model,
                agg,
                bm,
                features=[features_r, features_f] if bm == "mtl" else None,
            )
            optimizer.step()

            # measure accuracy and record loss
            prec1_retain = utils.accuracy(logits_r.data, retain_target)[0]

            loss_retain.update(ce_loss_r.item(), retain_image.size(0))
            top1_retain.update(prec1_retain.item(), retain_image.size(0))

             # measure accuracy and record loss
            prec1_forget = utils.accuracy(logits_f.data, forget_target)[0]

            # Track KL divergence for forget set (no longer tracking CE on forget set)
            loss_forget.update(kl_uniform_forget.item(), forget_image.size(0))
            top1_forget.update(prec1_forget.item(), forget_image.size(0))

            loss_entropy_forget.update(kl_uniform_forget.item(), forget_image.size(0))
            top1_entropy_forget.update(prec1_forget.item(), forget_image.size(0))

            # Track calibration loss on forget set
            loss_calibration.update(calib_loss.item(), forget_image.size(0))

            sum_loss = ce_loss_r + kl_uniform_forget + calib_loss
            loss_total.update(sum_loss.item(), retain_image.size(0) + forget_image.size(0))
            
            # Log step-level metrics to wandb (training phase - increment after)
            if wandb_run is not None:
                wandb_run.log({
                    "train/retain_ce_current": loss_retain.val,
                    "train/retain_ce_running": loss_retain.avg,
                    "train/retain_accuracy_current": top1_retain.val,
                    "train/retain_accuracy_running": top1_retain.avg,
                    "train/forget_kl_uniform_current": loss_forget.val,
                    "train/forget_kl_uniform_running": loss_forget.avg,
                    "train/forget_accuracy_current": top1_forget.val,
                    "train/forget_accuracy_running": top1_forget.avg,
                    "train/calibration_loss_current": loss_calibration.val,
                    "train/calibration_loss_running": loss_calibration.avg,
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
                    f"Forget KL-Uniform {loss_forget.val:.4f} ({loss_forget.avg:.4f})\t"
                    f"Forget Accuracy {top1_forget.val:.3f} ({top1_forget.avg:.3f})\t"
                    f"Calibration {loss_calibration.val:.4f} ({loss_calibration.avg:.4f})\t"
                    f"Total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t"
                    f"Time {end - start:.2f}"
                )
                start = time.time()
    else:
        for i in range(num_batches):
            retain_image, retain_target = next(retain_loader_iterator)
            forget_image, forget_target = next(forget_loader_iterator)
            retain_image, retain_target = retain_image.cuda(), retain_target.cuda()
            forget_image, forget_target = forget_image.cuda(), forget_target.cuda()

            # compute output for retained data
            if bm == "mtl":
                features_r = model.forward_features(retain_image)
                logits_r = model.forward_classifier(features_r)
                features_f = model.forward_features(forget_image)
                logits_f = model.forward_classifier(features_f)
            else:
                logits_r = model(retain_image)
                logits_f = model(forget_image)
            # compute cross-entropy loss for retained data
            ce_loss_retain = ce_criterion(logits_r, retain_target)

            # KL divergence to push forget set predictions toward uniform distribution
            kl_uniform_forget = kl_uniform_criterion(logits_f)

            # Calibration loss to reduce confidence on forget set
            calib_loss = calibration_loss(logits_f, forget_target)

            # compute total loss
            losses = [ce_loss_retain, kl_uniform_forget, calib_loss]
            # compute gradients
            optimizer.zero_grad()
            jd_backward(
                losses,
                model,
                agg,
                bm,
                features=[features_r, features_f] if bm == "mtl" else None,
            )
            # update model parameters
            optimizer.step()

            # measure accuracy and record loss for retained data
            prec1_retain = utils.accuracy(logits_r.data, retain_target)[0]

            loss_retain.update(ce_loss_retain.item(), retain_image.size(0))
            top1_retain.update(prec1_retain.item(), retain_image.size(0))

             # measure accuracy and record loss for forgotten data
            prec1_forget = utils.accuracy(logits_f.data, forget_target)[0]

            # Track KL divergence for forget set (no longer tracking CE on forget set)
            loss_forget.update(kl_uniform_forget.item(), forget_image.size(0))
            top1_forget.update(prec1_forget.item(), forget_image.size(0))

            loss_entropy_forget.update(kl_uniform_forget.item(), forget_image.size(0))
            top1_entropy_forget.update(prec1_forget.item(), forget_image.size(0))

            # Track calibration loss on forget set
            loss_calibration.update(calib_loss.item(), forget_image.size(0))

            # compute total loss
            sum_loss = ce_loss_retain + kl_uniform_forget + calib_loss
            # update total loss
            loss_total.update(sum_loss.item(), retain_image.size(0) + forget_image.size(0))
            
            # Log step-level metrics to wandb (training phase - increment after)
            if wandb_run is not None:
                wandb_run.log({
                    "train/retain_ce_current": loss_retain.val,
                    "train/retain_ce_running": loss_retain.avg,
                    "train/retain_accuracy_current": top1_retain.val,
                    "train/retain_accuracy_running": top1_retain.avg,
                    "train/forget_kl_uniform_current": loss_forget.val,
                    "train/forget_kl_uniform_running": loss_forget.avg,
                    "train/forget_accuracy_current": top1_forget.val,
                    "train/forget_accuracy_running": top1_forget.avg,
                    "train/calibration_loss_current": loss_calibration.val,
                    "train/calibration_loss_running": loss_calibration.avg,
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
                    f"Forget KL-Uniform {loss_forget.val:.4f} ({loss_forget.avg:.4f})\t"
                    f"Forget Accuracy {top1_forget.val:.3f} ({top1_forget.avg:.3f})\t"
                    f"Calibration {loss_calibration.val:.4f} ({loss_calibration.avg:.4f})\t"
                    f"Total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t"
                    f"Time {end - start:.2f}"
                )
                start = time.time()

    return total_steps, top1_retain.avg


@iterative_unlearn
def MO_JD_CE_UniDist_Calib(data_loaders, model, criterion, optimizer, epoch, args, total_steps=0, mask=None, wandb_run=None):
    return MO_JD_CE_UniDist_Calib_iter(
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
