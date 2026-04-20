from trainer import train

from .impl import iterative_unlearn


@iterative_unlearn
def retrain(data_loaders, model, criterion, optimizer, epoch, args, total_steps=0, mask=None, wandb_run=None):
    retain_loader = data_loaders["retain"]
    return train(retain_loader, model, criterion, optimizer, epoch, args, total_steps, mask, l1=False, wandb_run=wandb_run)
