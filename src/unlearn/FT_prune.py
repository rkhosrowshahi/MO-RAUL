import copy

import pruner
import trainer

from .FT import FT, FT_l1


def FT_prune(data_loaders, model, criterion, args, total_steps=1, mask=None, wandb_run=None):
    test_loader = data_loaders["test"]

    # save checkpoint
    initialization = copy.deepcopy(model.state_dict())

    # unlearn
    total_steps = FT_l1(data_loaders, model, criterion, args, total_steps=total_steps, mask=mask, wandb_run=wandb_run)

    # val
    pruner.check_sparsity(model)
    trainer.validate(test_loader, model, criterion, args)

    return total_steps, model
