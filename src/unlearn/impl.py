import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pruner
import torch
import utils
from pruner import extract_mask, prune_model_custom, remove_prune
from trainer import validate


def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f"{name}_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
    plt.close()


def save_unlearn_checkpoint(model, evaluation_result, args):
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    utils.save_checkpoint(state, False, args.save_dir, args.unlearn)
    utils.save_checkpoint(
        evaluation_result,
        False,
        args.save_dir,
        args.unlearn,
        filename="eval_result.pth.tar",
    )


def load_unlearn_checkpoint(model, device, args):
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn)
    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None

    current_mask = pruner.extract_mask(checkpoint["state_dict"])
    pruner.prune_model_custom(model, current_mask)
    pruner.check_sparsity(model)

    model.load_state_dict(checkpoint["state_dict"])

    # adding an extra forward process to enable the masks
    x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
    model.eval()
    with torch.no_grad():
        model(x_rand)

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args, mask=None, wandb_run=None, total_steps=1, **kwargs):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        if args.rewind_epoch != 0:
            initialization = torch.load(
                args.rewind_pth, map_location=torch.device("cuda:" + str(args.gpu))
            )
            current_mask = extract_mask(model.state_dict())
            remove_prune(model)
            # weight rewinding
            # rewind, initialization is a full model architecture without masks
            model.load_state_dict(initialization, strict=True)
            prune_model_custom(model, current_mask)
    
        optimizer = utils.get_optimizer(model, args, lr=args.unlearn_lr)
        # Align per-step warmup (trainer.train warmup_lr) with optimizer LR; see utils.warmup_lr.
        args.warmup_max_lr = args.unlearn_lr

        if args.imagenet_arch and args.unlearn == "retrain":
            lambda0 = (
                lambda cur_iter: (cur_iter + 1) / args.warmup
                if cur_iter < args.warmup
                else (
                    0.5
                    * (
                        1.0
                        + np.cos(
                            np.pi
                            * (
                                (cur_iter - args.warmup)
                                / (args.unlearn_epochs - args.warmup)
                            )
                        )
                    )
                )
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        elif str(getattr(args, "scheduler", None) or "").lower() == "none":
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )  # 0.1 is fixed
        if args.rewind_epoch != 0:
            # learning rate rewinding
            if scheduler is not None:
                for _ in range(args.rewind_epoch):
                    scheduler.step()

        best_val = float("-inf")

        # Use 0-based epochs like main_train so utils.warmup_lr(epoch, ...) matches pretrain.
        for epoch in range(args.unlearn_epochs):
            start_time = time.time()

            print(
                f"Epoch #{epoch + 1}, total_steps: {total_steps}, Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']:.12f}"
            )

            total_steps, train_acc = unlearn_iter_func(
                data_loaders, model, criterion, optimizer, epoch, args, total_steps=total_steps, mask=mask, wandb_run=wandb_run, **kwargs
            )

            if args.unlearn == "retrain":
                val_loader = data_loaders["val"]
                utils.dataset_convert_to_test(val_loader.dataset, args)
                val_acc = validate(val_loader, model, criterion, args)
                is_best = val_acc > best_val
                best_val = max(val_acc, best_val)
                state = {
                    "state_dict": model.state_dict(),
                    "evaluation_result": None,
                    "val_accuracy": val_acc,
                    "epoch": epoch + 1,
                }
                utils.save_checkpoint(
                    state,
                    is_SA_best=is_best,
                    save_path=args.save_dir,
                    pruning=args.unlearn,
                    filename="checkpoint.pth.tar",
                )

            if scheduler is not None:
                scheduler.step()

            epoch_duration = time.time() - start_time
            print("one epoch duration:{}".format(epoch_duration))

            # Log epoch-level metrics to wandb using cumulative step count (evaluation phase - no increment)
            log_payload = {
                "train/epoch": epoch + 1,
                "train/learning_rate": optimizer.state_dict()["param_groups"][0]["lr"],
                "train/epoch_duration": epoch_duration,
                "train/train_accuracy": train_acc,
            }
            if args.unlearn == "retrain":
                log_payload["train/val_accuracy"] = val_acc
                log_payload["train/best_val_accuracy"] = best_val
            if wandb_run is not None:
                wandb_run.log(log_payload, step=total_steps)

        if args.unlearn == "retrain":
            best_path = os.path.join(
                args.save_dir, str(args.unlearn) + "model_SA_best.pth.tar"
            )
            if os.path.isfile(best_path):
                device = next(model.parameters()).device
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["state_dict"])
                print(
                    "Retrain: loaded best val checkpoint | val_acc={:.3f} epoch={}".format(
                        ckpt.get("val_accuracy", float("nan")),
                        ckpt.get("epoch", "?"),
                    )
                )

        return total_steps

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)
