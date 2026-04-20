import sys
import time
import itertools
import torch
import torch.nn as nn
from torchjd.aggregation import PCGrad, UPGrad, AlignedMTL, DualProj, IMTLG, MGDA, Mean, CAGrad, NashMTL
import utils

from ..impl import iterative_unlearn
from .utils import assert_split_forward_if_mtl, jd_backward

sys.path.append(".")
from imagenet import get_x_y_from_data_dict


def kl_uniform_criterion(logits):
    """
    Cross-entropy H(U, P) between uniform U (1/C per class) and model softmax P.

    Minimizing this equals minimizing KL(U || P) up to the constant H(U) = log(C),
    so it encourages high-entropy (near-uniform) predictions on the forget batch.

    This is not KL(P || U): that would be sum_c p_c (log p_c + log C).
    """
    num_classes = logits.size(1)
    probs = torch.softmax(logits, dim=1)
    uniform_dist = torch.ones_like(probs) / num_classes
    # H(U,P) = -sum_c u_c log p_c with u_c = 1/C
    loss = torch.mean(torch.sum(-uniform_dist * torch.log(probs), dim=1))
    return loss


def MO_JD_iter(
    data_loaders, model, criterion, optimizer, epoch, args, total_steps=0, aggregator=None, mask=None, wandb_run=None
):
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]

    loss_retain = utils.AverageMeter()
    top1_retain = utils.AverageMeter()
    loss_forget = utils.AverageMeter()
    top1_forget = utils.AverageMeter()
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
    num_batches = len(forget_loader)
    if num_batches < args.print_freq:
        args.print_freq = num_batches // 2

    start = time.time()
    if args.imagenet_arch:
        for i in range(num_batches):
            retain_image, retain_target = next(retain_loader_iterator)
            forget_image, forget_target = next(forget_loader_iterator)
            retain_image, retain_target = retain_image.cuda(), retain_target.cuda()
            forget_image, forget_target = forget_image.cuda(), forget_target.cuda()

            # extract features for retained data
            features_r = model.forward_features(retain_image)
            # compute output for retained data
            logits_r = model.forward_classifier(features_r)
            ce_loss_r = ce_criterion(logits_r, retain_target)

            # extract features for forgotten data
            features_f = model.forward_features(forget_image)
            # compute output for forgotten data
            logits_f = model.forward_classifier(features_f)
            # Forget: minimize H(U,P) (same as KL(U||P) up to log C); see kl_uniform_criterion
            kl_uniform_forget = kl_uniform_criterion(logits_f)

            losses = [ce_loss_r, kl_uniform_forget]

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

            # Track forget UKL term (H(U,P)); no CE on forget
            loss_forget.update(kl_uniform_forget.item(), forget_image.size(0))
            top1_forget.update(prec1_forget.item(), forget_image.size(0))

            sum_loss = ce_loss_r + kl_uniform_forget
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

            # extract features for retained data
            features_r = model.forward_features(retain_image)
            # compute output for retained data
            logits_r = model.forward_classifier(features_r)
            # compute cross-entropy loss for retained data
            ce_loss_retain = ce_criterion(logits_r, retain_target)

            # extract features for forgotten data
            features_f = model.forward_features(forget_image)
            # compute output for forgotten data
            logits_f = model.forward_classifier(features_f)
            # Forget: minimize H(U,P) (same as KL(U||P) up to log C); see kl_uniform_criterion
            kl_uniform_forget = kl_uniform_criterion(logits_f)

            # compute total loss
            losses = [ce_loss_retain, kl_uniform_forget]
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

            # Track forget UKL term (H(U,P)); no CE on forget
            loss_forget.update(kl_uniform_forget.item(), forget_image.size(0))
            top1_forget.update(prec1_forget.item(), forget_image.size(0))

            # compute total loss
            sum_loss = ce_loss_retain + kl_uniform_forget
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
                    f"Total Loss {loss_total.val:.4f} ({loss_total.avg:.4f})\t"
                    f"Time {end - start:.2f}"
                )
                start = time.time()

    return total_steps, top1_retain.avg


@iterative_unlearn
def MO_JD_CE_UniDist(data_loaders, model, criterion, optimizer, epoch, args, total_steps=0, mask=None, wandb_run=None):
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
