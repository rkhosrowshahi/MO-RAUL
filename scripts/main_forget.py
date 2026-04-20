import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import copy
import os
from collections import OrderedDict

import numpy as np

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.utils.data
import unlearn
import utils
import wandb
from trainer import validate
from unlearn.evomoul.evaluation import accuracy_metric, evaluate_metrics, mia_metric_as_percent

def main():
    args = arg_parser.parse_args()
    
    # Ensure save_dir exists and is absolute before initializing wandb
    if args.save_dir:
        args.save_dir = os.path.abspath(args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)

    print(f"save_dir: {args.save_dir}")
    
    # Initialize wandb
    wandb_config = {
        "project": args.wandb_project,
        "config": vars(args),
        "mode": args.wandb_mode,
        "dir": args.save_dir,
    }
    
    # Set run name
    if args.wandb_name:
        wandb_config["name"] = args.wandb_name
    elif args.wandb_run_name:
        wandb_config["name"] = args.wandb_run_name
    else:
        arch_str = getattr(args, 'arch', 'unknown')
        wandb_config["name"] = f"{args.unlearn}_{args.dataset}_{arch_str}"
    
    # Set entity if provided
    if args.wandb_entity:
        wandb_config["entity"] = args.wandb_entity

    # Set group if provided (W&B max group length is 128)
    if args.wandb_group:
        _g = utils.wandb_sanitize_group(args.wandb_group)
        if _g != args.wandb_group:
            print(f"wandb group truncated for W&B 128-char limit: {_g!r}")
        wandb_config["group"] = _g
    
    # Set tags
    default_tags = [args.unlearn, args.dataset]
    if hasattr(args, 'arch'):
        default_tags.append(args.arch)
    wandb_config["tags"] = default_tags + args.wandb_tags
    
    wandb.init(**wandb_config)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    
    # Match upstream: optional global setup_seed (--seed); CIFAR-10 model init uses --train_seed in utils.
    if args.seed:
        utils.setup_seed(args.seed)

    print(f"args.no_aug: {args.no_aug}")
    
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()
    
    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        return utils.get_loader_from_dataset(
            dataset, batch_size=batch_size, seed=seed, shuffle=shuffle
        )

    _t = getattr(marked_loader.dataset, "targets", None)
    if _t is None:
        _t = getattr(marked_loader.dataset, "labels", None)
    if _t is not None:
        _t = np.asarray(_t)
        _n_forget = int((_t < 0).sum())
        _n_retain = int((_t >= 0).sum())
        if _n_forget == 0 or _n_retain == 0:
            raise ValueError(
                f"Invalid forget/retain partition: n_forget={_n_forget}, n_retain={_n_retain}, "
                f"n_total={len(_t)}. Set num_indexes_to_replace (random forgetting), "
                "class_to_replace in [0, num_classes-1], or indexes_to_replace. "
                "With class_to_replace=-1 you must set num_indexes_to_replace."
            )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=args.seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=args.seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=args.seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=args.seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=args.seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=args.seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    utils.assert_forget_retain_partition_indices(marked_loader.dataset)

    # forget / forget_loader: true labels — used for metrics, MIA, and full-dataset eval.
    # forget_with_random_label: random labels (copy) — EvoMOUL objectives that train vs random forget targets.
    forget_with_random_label_loader = None
    _mo = getattr(args, "mo_type", None)
    if _mo in (
        "retain_ce-forget_random_label_ce",
        "retain_ce-forget_random_label_f1",
        "retain_ce-retain_neg_f1-forget_random_label_ce-forget_random_label_neg_f1",
    ):
        forget_dataset_mo = copy.deepcopy(forget_dataset)
        utils.apply_fixed_random_labels_to_forget_dataset(
            forget_dataset_mo, args.num_classes, args.seed
        )
        forget_with_random_label_loader = replace_loader_dataset(
            forget_dataset_mo, seed=args.seed, shuffle=True
        )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )
    if forget_with_random_label_loader is not None:
        unlearn_data_loaders["forget_with_random_label"] = forget_with_random_label_loader

    criterion = nn.CrossEntropyLoss(
        label_smoothing=getattr(args, "label_smoothing", 0.0)
    )
    evaluation_result = None
    total_steps = 0  # For evaluation-phase wandb step when unlearn was not run (e.g. resume)

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        # Load weights BEFORE base evaluation (was previously done after, causing ~9% random acc)
        if args.unlearn != "retrain" and "state_dict" in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        if "evaluation_result" not in checkpoint.keys():
            evaluation_result = {}
            accuracy = {}
            for name, loader in unlearn_data_loaders.items():
                if name == "forget_with_random_label":
                    continue
                utils.dataset_convert_to_test(loader.dataset, args)
                val_acc = validate(loader, model, criterion, args)
                accuracy[name] = val_acc
                print(f"{name} acc: {val_acc}")

            evaluation_result["accuracy"] = accuracy
            test_len = len(test_loader.dataset)
            forget_len = len(forget_dataset)
            retain_len = len(retain_dataset)

            utils.dataset_convert_to_test(retain_dataset, args)
            utils.dataset_convert_to_test(forget_loader, args)
            utils.dataset_convert_to_test(test_loader, args)

            shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
            shadow_train_loader = torch.utils.data.DataLoader(
                shadow_train, batch_size=args.batch_size, shuffle=False
            )

            evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
                shadow_train=shadow_train_loader,
                shadow_test=test_loader,
                target_train=None,
                target_test=forget_loader,
                model=model,
            )

            checkpoint["evaluation_result"] = evaluation_result
            torch.save(checkpoint, args.model_path)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        unlearn_result = unlearn_method(unlearn_data_loaders, model, criterion, args, wandb_run=wandb)
        unlearn.save_unlearn_checkpoint(model, None, args)
        # Extract total_steps for evaluation-phase wandb logging
        total_steps = (
            unlearn_result if isinstance(unlearn_result, int) else
            (unlearn_result[0] if isinstance(unlearn_result, tuple) and len(unlearn_result) >= 1 else 0)
        )

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            if name == "forget_with_random_label":
                continue
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        evaluation_result["accuracy"] = accuracy
        
        # Log final accuracies to wandb (evaluation phase - no increment)
        wandb.log({
            "final/retain_accuracy": accuracy.get("retain", 0),
            "final/forget_accuracy": accuracy.get("forget", 0),
            "final/unlearn_accuracy": 100 - accuracy.get("forget", 0),
            "final/val_accuracy": accuracy.get("val", 0),
            "final/test_accuracy": accuracy.get("test", 0),
        }, step=total_steps)
        
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    if "SVC_MIA_forget_efficacy" not in evaluation_result:
        test_len = len(test_loader.dataset)
        forget_len = len(forget_dataset)
        retain_len = len(retain_dataset)

        utils.dataset_convert_to_test(retain_dataset, args)
        utils.dataset_convert_to_test(forget_loader, args)
        utils.dataset_convert_to_test(test_loader, args)

        shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train, batch_size=args.batch_size, shuffle=False
        )

        evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
            shadow_train=shadow_train_loader,
            shadow_test=test_loader,
            target_train=None,
            target_test=forget_loader,
            model=model,
        )
        print(f"MIA: {evaluation_result['SVC_MIA_forget_efficacy']}")
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
    else:
        print(f"MIA: {evaluation_result['SVC_MIA_forget_efficacy']}")

    # Gap to retrain (when retrain_path is available; Retrain baseline has gap=0)
    gap_log = {}
    if getattr(args, "retrain_path", None):
        retrain_results = evaluate_metrics(args.retrain_path)
        if retrain_results is not None:
            acc = evaluation_result.get("accuracy", {})
            retain_acc = acc.get("retain")
            forget_acc = acc.get("forget")
            unlearn_acc = 100 - forget_acc if forget_acc is not None else None
            test_acc = acc.get("test")
            mia_result = evaluation_result.get("SVC_MIA_forget_efficacy")
            curr_mia = mia_result.get("confidence") if isinstance(mia_result, dict) else (mia_result if isinstance(mia_result, (int, float)) else None)
            rr_r = accuracy_metric(retrain_results, "retain")
            rr_f = accuracy_metric(retrain_results, "forget")
            rr_u = accuracy_metric(retrain_results, "unlearn")
            rr_t = accuracy_metric(retrain_results, "test")
            if retain_acc is not None and rr_r is not None:
                gap_log["final/gap_retain_accuracy"] = abs(retain_acc - rr_r)
            if forget_acc is not None and rr_f is not None:
                gap_log["final/gap_forget_accuracy"] = abs(forget_acc - rr_f)
            if unlearn_acc is not None and rr_u is not None:
                gap_log["final/gap_unlearn_accuracy"] = abs(unlearn_acc - rr_u)
            if test_acc is not None and rr_t is not None:
                gap_log["final/gap_test_accuracy"] = abs(test_acc - rr_t)
            retrain_results_mia = retrain_results.get("confidence")
            curr_mia_p = mia_metric_as_percent(curr_mia)
            retrain_mia_p = mia_metric_as_percent(retrain_results_mia)
            if curr_mia_p == curr_mia_p and retrain_mia_p == retrain_mia_p:
                gap_log["final/gap_mia"] = abs(curr_mia_p - retrain_mia_p)
            # Mean of |UA|, |RA|, |TA|, |MIA| vs retrain (same as eval/average_gap; forget gap is redundant with UA)
            _avg_gap_keys = (
                "final/gap_unlearn_accuracy",
                "final/gap_retain_accuracy",
                "final/gap_test_accuracy",
                "final/gap_mia",
            )
            valid_gaps = [
                gap_log[k]
                for k in _avg_gap_keys
                if k in gap_log and gap_log[k] is not None and gap_log[k] == gap_log[k]
            ]
            if valid_gaps:
                gap_log["final/average_gap"] = sum(valid_gaps) / len(valid_gaps)
    elif args.unlearn == "retrain":
        # Retrain is the baseline; gap to self is 0
        gap_log = {
            "final/gap_retain_accuracy": 0.0,
            "final/gap_forget_accuracy": 0.0,
            "final/gap_unlearn_accuracy": 0.0,
            "final/gap_test_accuracy": 0.0,
            "final/gap_mia": 0.0,
            "final/average_gap": 0.0,
        }

    # Ensure wandb finishes even on interruption
    try:
        final_log = {f"final/{k}": v for k, v in evaluation_result.items()}
        final_log.update(gap_log)
        _svc_mia = evaluation_result.get("SVC_MIA_forget_efficacy")
        if isinstance(_svc_mia, dict) and "confidence" in _svc_mia:
            final_log["final/mia_accuracy"] = _svc_mia["confidence"]
        wandb.log(final_log, step=total_steps)
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
    except KeyboardInterrupt:
        print("\n  Experiment interrupted by user")
        # Log interruption to wandb
        try:
            wandb.log({"status": "interrupted"}, step=total_steps)
        except:
            pass
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        # Log error to wandb
        try:
            wandb.log({"status": "error", "error_message": str(e)}, step=total_steps)
        except:
            pass
        raise
    finally:
        # Always finish wandb run to ensure logs are synced
        try:
            wandb.finish()
        except:
            pass


if __name__ == "__main__":
    main()