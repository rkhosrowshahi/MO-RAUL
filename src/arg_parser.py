import argparse
import sys
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Classification of SalUn Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        "--data", type=str, default="../data", help="location of the data corpus"
    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument(
        "--input_size", type=int, default=32, help="size of input images"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./datasets/tiny-imagenet-200",
        help="dir to tiny-imagenet",
    )
    parser.add_argument("--num_classes", type=int, default=10)

    ##################################### Architecture ############################################
    parser.add_argument(
        "--arch", type=str, default="resnet18", help="model architecture"
    )
    parser.add_argument(
        "--imagenet_arch",
        action="store_true",
        help="architecture for imagenet size samples",
    )
    parser.add_argument(
        "--train_y_file",
        type=str,
        default="./labels/train_ys.pth",
        help="labels for training files",
    )
    parser.add_argument(
        "--val_y_file",
        type=str,
        default="./labels/val_ys.pth",
        help="labels for validation files",
    )

    ##################################### General setting ############################################
    parser.add_argument(
        "--seed",
        default=2,
        type=int,
        help="random seed for data splits / marking forget indices (passed to dataloaders as in upstream SalUn)",
    )
    parser.add_argument(
        "--train_seed",
        default=1,
        type=int,
        help="seed for training-time RNG (model init on CIFAR-10: setup_seed before building the model; "
        "defaults match upstream Unlearn-Saliency: seed=2, train_seed=1)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument(
        "--workers", type=int, default=0, help="number of workers in dataloader"
    )
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument(
        "--save_dir",
        help="The directory used to save the trained models",
        default=None,
        type=str,
    )
    parser.add_argument("--model_path", type=str, default=None, help="the path of original model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="sgd", 
        choices=["sgd", "adam", "adamw"],
        help="optimizer type (sgd, adam, adamw)"
    )
    parser.add_argument(
        "--adam_beta1", 
        default=0.9, 
        type=float, 
        help="beta1 parameter for Adam/AdamW optimizer"
    )
    parser.add_argument(
        "--adam_beta2", 
        default=0.999, 
        type=float, 
        help="beta2 parameter for Adam/AdamW optimizer"
    )
    parser.add_argument(
        "--epochs", default=182, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument(
        "--warmup_start_lr",
        default=None,
        type=float,
        help="starting LR for warmup (linear warmup from this to lr). If None, uses lr/warmup.",
    )
    parser.add_argument(
        "--scheduler",
        default=None,
        type=str,
        choices=["cosine", "step", "none"],
        help="scheduler type for imagenet_arch: cosine or step; use 'none' for unlearning with fixed unlearn_lr (no stepping).",
    )
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument(
        "--label_smoothing",
        default=0.0,
        type=float,
        help="label smoothing for CrossEntropyLoss (e.g. 0.1 for Swin)",
    )
    parser.add_argument(
        "--no-aug",
        action="store_true",
        default=False,
        help="No augmentation in training dataset (transformation).",
    )
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")

    ##################################### Pruning setting #################################################
    parser.add_argument("--prune", type=str, default="omp", help="method to prune")
    parser.add_argument(
        "--pruning_times",
        default=1,
        type=int,
        help="overall times of pruning (only works for IMP)",
    )
    parser.add_argument(
        "--rate", default=0.95, type=float, help="pruning rate"
    )  # pruning rate is always 20%
    parser.add_argument(
        "--prune_type",
        default="rewind_lt",
        type=str,
        help="IMP type (lt, pt or rewind_lt)",
    )
    parser.add_argument(
        "--random_prune", action="store_true", help="whether using random prune"
    )
    parser.add_argument("--rewind_epoch", default=0, type=int, help="rewind checkpoint")
    parser.add_argument(
        "--rewind_pth", default=None, type=str, help="rewind checkpoint to load"
    )

    ##################################### Unlearn setting #################################################
    parser.add_argument(
        "--unlearn", type=str, default="retrain", help="method to unlearn"
    )
    parser.add_argument(
        "--mo_aggregator",
        type=str,
        default="upgrad",
        choices=[
            "mean",
            "pcgrad",
            "upgrad",
            "amtl",
            "mgda",
            "cagrad",
            "dualproj",
            "imtl",
            "nashmtl",
        ],
        help="TorchJD aggregation for MO_JD_CE_CE (dual CE), MO_JD_CE_KL, MO_JD_CE_UniDist, and MO_JD_3M.",
    )
    parser.add_argument(
        "--backward",
        type=str,
        default="full",
        choices=["full", "mtl"],
        help="MOJD TorchJD step (MO_JD_CE_CE, MO_JD_CE_KL, MO_JD_CE_UniDist, MO_JD_3M). "
        "'full': backward + jac_to_grad on all parameters. "
        "'mtl': mtl_backward + jac_to_grad on encoder from forward_features (requires split model).",
    )
    parser.add_argument(
        "--unlearn_lr", default=0.01, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--unlearn_epochs",
        default=10,
        type=int,
        help="number of total epochs for unlearn to run",
    )

    parser.add_argument(
        "--num_indexes_to_replace",
        type=int,
        default=None,
        help="Number of data to forget",
    )
    parser.add_argument(
        "--class_to_replace",
        type=int,
        default=None,
        help="Class id to forget, or -1 with num_indexes_to_replace for random subset. "
        "Default None: use num_indexes_to_replace alone for random forgetting, or pass indexes_to_replace.",
    )

    parser.add_argument(
        "--indexes_to_replace",
        type=list,
        default=None,
        help="Specific index data to forget",
    )
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise")
    parser.add_argument("--mask_path", default=None, type=str, help="the path of saliency map")

    ##################################### Wandb setting #################################################
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="mu-classification", 
        help="wandb project name"
    )
    parser.add_argument(
        "--wandb_run_name", 
        type=str, 
        default=None, 
        help="wandb run name (defaults to method_dataset_arch)"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="alias for wandb run name"
    )
    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        default=None, 
        help="wandb entity/team name"
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="wandb group name for grouping related runs"
    )
    parser.add_argument(
        "--wandb_mode", 
        type=str, 
        default="online", 
        choices=["online", "offline", "disabled"],
        help="wandb mode: online, offline, or disabled"
    )
    parser.add_argument(
        "--wandb_tags", 
        type=str, 
        nargs="+", 
        default=[], 
        help="additional tags for wandb run"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (values override defaults; CLI overrides config)",
    )

    args, _ = parser.parse_known_args()
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config:
            valid_dests = {a.dest for a in parser._actions if a.dest != "help"}
            config_key_aliases = {}

            def apply_config(key, val):
                dest = key.replace("-", "_")
                dest = config_key_aliases.get(dest, dest)
                if isinstance(val, dict):
                    for k, v in val.items():
                        nested_dest = f"{dest}_{k}".replace("-", "_")
                        if nested_dest in valid_dests:
                            parser.set_defaults(**{nested_dest: v})
                elif dest in valid_dests:
                    parser.set_defaults(**{dest: val})

            for key, value in config.items():
                apply_config(key, value)
        # CLI still wins over YAML defaults
        args = parser.parse_args(sys.argv[1:])
    else:
        args = parser.parse_args()
    return args
