# -*- coding: utf-8 -*-
"""
Modified from fairchem: https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/_cli.py
To support NERSC Slurm job submission

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import errno
import logging
import os
import sys
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
import torch.distributed
from submitit import AutoExecutor
from submitit.helpers import Checkpointable, DelayedSubmission
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from torch.utils.data import BatchSampler, Dataset, DistributedSampler
from typing_extensions import override

if TYPE_CHECKING:
    import argparse

from fairchem.core.common import distutils, gp_utils
from fairchem.core.common.data_parallel import (
    StatefulDistributedSampler,
    _balanced_partition,
    _ensure_supported,
)
from fairchem.core.common.flags import flags
from fairchem.core.common.logger import Logger, WandBLogger
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import (
    build_config,
    create_grid,
    load_state_dict,
    match_state_dict,
    new_trainer_context,
    save_experiment_log,
    setup_logging,
)
from fairchem.core.models.equiformer_v2.trainers.forces_trainer import (
    ExponentialMovingAverage,
    LRScheduler,
)
from fairchem.core.modules.normalization.element_references import (
    create_element_references,
)
from fairchem.core.modules.normalization.normalizer import create_normalizer
from fairchem.core.modules.scaling.compat import load_scales_compat
from fairchem.core.trainers.ocp_trainer import OCPTrainer

import wandb  # isort:skip

# Add parent directory to path for imports
sys.path.append("../")

import ase
from fairchem.core.datasets.base_dataset import BaseDataset

# set the cuda home
from fairchem.core.modules.transforms import DataTransforms
from torch.nn.parallel.distributed import DistributedDataParallel
from torch_geometric.data import Data


@registry.register_dataset("spice_xyz")
class SPICExyzDataset(BaseDataset):
    def __init__(self, config, transform=None) -> None:
        super().__init__(config)
        assert (
            len(self.paths) == 1
        ), f"{type(self)} does not support a list of src paths."
        self.path = self.paths[0]

        self.atoms_list = ase.io.read(self.path, index=":")

        self.atom_reference = np.array(
            [
                0.00000000e00,
                -1.63841057e01,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -1.03758911e03,
                -1.49072485e03,
                -2.04814404e03,
                -2.71846191e03,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -9.29155469e03,
                -1.08369053e04,
                -1.25243545e04,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -7.00463750e04,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                -8.10290869e03,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ]
        )
        self.num_samples = len(self.atoms_list)
        self.transforms = DataTransforms(self.config.get("transforms", {}))
        self.subset_name2id = {
            "PubChem": 0,  # 34093,
            "DES370K Monomers": 1,  # 889,
            "DES370K Dimers": 2,  # 13908,
            "Dipeptides": 3,  # 1025,
            "Solvated Amino Acids": 4,  # 52
            "water": 5,  # 84,
            "QMugs": 6,  # 144,
        }

    def __getitem__(self, idx):
        energy = self.atoms_list[idx].get_potential_energy()
        forces = self.atoms_list[idx].get_forces()
        atomic_numbers = self.atoms_list[idx].get_atomic_numbers()
        pos = self.atoms_list[idx].get_positions()
        subset_name = self.atoms_list[idx].info["config_type"]

        unique, counts = np.unique(atomic_numbers, return_counts=True)
        energy = energy - np.sum(self.atom_reference[unique] * counts)

        out = Data(
            **{
                "energy": energy,
                "forces": torch.from_numpy(forces),
                "atomic_numbers": torch.from_numpy(atomic_numbers),
                "pos": torch.from_numpy(pos),
                "natoms": len(atomic_numbers),
                "subset_name": self.subset_name2id[subset_name],
                "forces_subset_name": torch.zeros(len(atomic_numbers))
                + self.subset_name2id[subset_name],
                "fixed": torch.zeros(len(atomic_numbers)),
                "pbc": torch.zeros(3),
                "cell": torch.zeros(1, 3, 3),
            }
        )
        # print(out)
        return self.transforms(out)

    def __len__(self) -> int:
        return self.num_samples


@registry.register_loss("rmse")
class RMSELoss(torch.nn.Module):
    """
    Currently this loss is intened to used with vectors.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        # print("energy rmse  and mae is :",pred.shape,torch.sqrt(torch.mean((pred - target)**2)),torch.mean(torch.abs(pred - target)))
        return torch.sqrt(torch.mean((pred - target) ** 2))


@registry.register_trainer("new_equiformerv2_forces")
class New_EquiformerV2ForcesTrainer(OCPTrainer):
    # This trainer does a few things differently from the parent forces trainer:
    # - Support for cosine LR scheduler.
    # - When using the LR scheduler, it first converts the epochs into number of
    #   steps and then passes it to the scheduler. That way in the config
    #   everything can be specified in terms of epochs.

    def load_model(self) -> None:
        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']['name']}")

        model_config_copy = copy.deepcopy(self.config["model"])
        model_name = model_config_copy.pop("name")
        self.model = registry.get_model_class(model_name)(
            **model_config_copy,
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{num_params} parameters."
            )

        if self.logger is not None:
            # only "watch" model if user specify watch: True because logging gradients
            # spews too much data into W&B and makes the UI slow to respond
            if "watch" in self.config["logger"]:
                self.logger.watch(
                    self.model, log_freq=int(self.config["logger"]["watch"])
                )
            self.logger.log_summary({"num_params": num_params})

        # NOTICE: # change this to be True from the original FairChemCode
        if distutils.initialized():
            self.model = DistributedDataParallel(
                self.model,
                find_unused_parameters=True,  # change this to be True from the original FairChemCode
            )

    def load_extras(self) -> None:
        def multiply(obj, num):
            if isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = obj[i] * num
            else:
                obj = obj * num
            return obj

        self.config["optim"]["scheduler_params"]["epochs"] = self.config["optim"][
            "max_epochs"
        ]
        self.config["optim"]["scheduler_params"]["lr"] = self.config["optim"][
            "lr_initial"
        ]

        # convert epochs into number of steps
        if self.train_loader is None:
            logging.warning("Skipping scheduler setup. No training set found.")
            self.scheduler = None
        else:
            n_iter_per_epoch = len(self.train_loader)
            scheduler_params = self.config["optim"]["scheduler_params"]
            for k in scheduler_params:
                if "epochs" in k:
                    if isinstance(scheduler_params[k], (int, float)):
                        scheduler_params[k] = int(
                            multiply(scheduler_params[k], n_iter_per_epoch)
                        )
                    elif isinstance(scheduler_params[k], list):
                        scheduler_params[k] = [
                            int(x)
                            for x in multiply(scheduler_params[k], n_iter_per_epoch)
                        ]
            self.scheduler = LRScheduler(self.optimizer, self.config["optim"])

        self.clip_grad_norm = self.config["optim"].get("clip_grad_norm")
        self.ema_decay = self.config["optim"].get("ema_decay")
        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None

    def get_sampler(
        self, dataset, batch_size: int, shuffle: bool
    ) -> New_BalancedBatchSampler:
        balancing_mode = self.config["optim"].get("load_balancing", None)
        on_error = self.config["optim"].get("load_balancing_on_error", None)
        if balancing_mode is not None:
            if on_error is None:
                on_error = "raise"
        else:
            balancing_mode = "atoms"

        if on_error is None:
            on_error = "warn_and_no_balance"

        if gp_utils.initialized():
            num_replicas = gp_utils.get_dp_world_size()
            rank = gp_utils.get_dp_rank()
        else:
            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()
        return New_BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            on_error=on_error,
            seed=self.config["cmd"]["seed"],
        )

    def load_checkpoint(
        self,
        checkpoint_path: str,
        checkpoint: dict | None = None,
        inference_only: bool = False,
    ) -> None:
        map_location = torch.device("cpu") if self.cpu else self.device
        if checkpoint is None:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    errno.ENOENT, "Checkpoint file not found", checkpoint_path
                )
            logging.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # attributes that are necessary for training and validation
        if inference_only is False:
            self.epoch = checkpoint.get("epoch", 0)
            self.step = checkpoint.get("step", 0)
            self.best_val_metric = checkpoint.get("best_val_metric", None)
            self.primary_metric = checkpoint.get("primary_metric", None)
            print(
                "loaded epoch step, ***:",
                self.epoch,
                self.step,
                self.best_val_metric,
                self.primary_metric,
            )
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            else:
                print("we can not load optimizer from ckpt")
            if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
                self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
            else:
                print("we can not load scheduler from ckpt")
        else:
            logging.info(
                "Loading checkpoint in inference-only mode, not loading keys associated with trainer state!"
            )

        if "ema" in checkpoint and checkpoint["ema"] is not None and self.ema:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None

        new_dict = match_state_dict(self.model.state_dict(), checkpoint["state_dict"])
        strict = self.config.get("task", {}).get("strict_load", True)
        load_state_dict(self.model, new_dict, strict=strict)

        scale_dict = checkpoint.get("scale_dict", None)
        if scale_dict:
            logging.info(
                "Overwriting scaling factors with those loaded from checkpoint. "
                "If you're generating predictions with a pretrained checkpoint, this is the correct behavior. "
                "To disable this, delete `scale_dict` from the checkpoint. "
            )
            load_scales_compat(self._unwrapped_model, scale_dict)

        for key, state_dict in checkpoint["normalizers"].items():
            ### Convert old normalizer keys to new target keys
            if key == "target":
                target_key = "energy"
            elif key == "grad_target":
                target_key = "forces"
            else:
                target_key = key

            if target_key not in self.normalizers:
                self.normalizers[target_key] = create_normalizer(state_dict=state_dict)
            else:
                mkeys = self.normalizers[target_key].load_state_dict(state_dict)
                assert len(mkeys.missing_keys) == 0
                assert len(mkeys.unexpected_keys) == 0

            self.normalizers[target_key].to(map_location)

        for key, state_dict in checkpoint.get("elementrefs", {}).items():
            if key not in self.elementrefs:
                self.elementrefs[key] = create_element_references(state_dict=state_dict)
            else:
                mkeys = self.elementrefs[key].load_state_dict(state_dict)
                assert len(mkeys.missing_keys) == 0
                assert len(mkeys.unexpected_keys) == 0

            self.elementrefs[key].to(map_location)

        if self.scaler and checkpoint["amp"]:
            self.scaler.load_state_dict(checkpoint["amp"])

    def _compute_metrics(self, out, batch, evaluator, metrics=None):
        if metrics is None:
            metrics = {}
        # this function changes the values in the out dictionary,
        # make a copy instead of changing them in the callers version
        out = {k: v.clone() for k, v in out.items()}

        natoms = batch.natoms
        batch_size = natoms.numel()

        ### Retrieve free atoms
        if "fixed" in batch:
            fixed = batch.fixed
        else:
            fixed = torch.zeros_like(
                batch.batch, dtype=torch.bool, device=batch.batch.device
            )
        mask = fixed == 0

        s_idx = 0
        natoms_free = []
        for _natoms in natoms:
            natoms_free.append(torch.sum(mask[s_idx : s_idx + _natoms]).item())
            s_idx += _natoms
        natoms = torch.LongTensor(natoms_free).to(self.device)

        targets = {}
        for target_name in self.output_targets:
            target = batch[target_name]
            num_atoms_in_batch = batch.natoms.sum()

            if (
                self.output_targets[target_name]["level"] == "atom"
                and self.output_targets[target_name]["eval_on_free_atoms"]
            ):
                target = target[mask]
                out[target_name] = out[target_name][mask]
                num_atoms_in_batch = natoms.sum()

            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_name]["level"] == "atom":
                target = target.view(num_atoms_in_batch, -1)
            else:
                target = target.view(batch_size, -1)

            out[target_name] = self._denorm_preds(target_name, out[target_name], batch)
            targets[target_name] = target

        targets["natoms"] = natoms
        out["natoms"] = natoms
        # raise ValueError
        # # add all other tensor properties too, but filter out the ones that are changed above
        # for key in filter(
        #     lambda k: k not in [*list(self.output_targets.keys()), "natoms"]
        #     and isinstance(batch[k], torch.Tensor),
        #     batch.keys(),
        # ):
        #     targets[key] = batch[key].to(self.device)
        #     out[key] = targets[key]
        # # print("energy:",out["energy"],targets["energy"])
        # # print("forces:",out["forces"].shape,natoms,out["forces"],targets["forces"])
        # # print("mae forces:",torch.mean(torch.abs(out["forces"])),torch.mean(torch.abs(targets["forces"])))
        # # print("cos:",torch.mean(torch.cosine_similarity(out["forces"],targets["forces"],dim = 1)))
        if (
            self.model.training is False
            and "subset_name" in batch
            and batch.subset_name is not None
        ):
            metrics = evaluator.eval(out, targets, prev_metrics=metrics)
            for subset_id in batch.subset_name.unique():
                e_mask = batch.subset_name == subset_id
                f_mask = batch.forces_subset_name == subset_id
                tmp_natoms = natoms[e_mask]
                tmp_out = {
                    "energy": out["energy"][e_mask] / tmp_natoms.float(),
                    "forces": out["forces"][f_mask],
                    "natoms": tmp_natoms,
                }
                tmp_targets = {
                    "energy": targets["energy"][e_mask] / tmp_natoms.float(),
                    "forces": targets["forces"][f_mask],
                    "natoms": tmp_natoms,
                }
                tmp_metric = evaluator.eval(tmp_out, tmp_targets, prev_metrics={})
                for key in tmp_metric:
                    metric_name = key + "_" + str(subset_id.item())
                    if metric_name not in metrics:
                        metrics[metric_name] = {
                            "metric": None,
                            "total": 0,
                            "numel": 0,
                        }
                    # If dictionary, we expect it to have `metric`, `total`, `numel`.
                    metrics[metric_name]["total"] += tmp_metric[key]["total"]
                    metrics[metric_name]["numel"] += tmp_metric[key]["numel"]
                    metrics[metric_name]["metric"] = (
                        metrics[metric_name]["total"] / metrics[metric_name]["numel"]
                    )
        else:
            metrics = evaluator.eval(out, targets, prev_metrics=metrics)

        return metrics


class New_BalancedBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        device: torch.device,
        seed: int,
        mode: bool | Literal["atoms"] = "atoms",
        shuffle: bool = True,
        on_error: Literal["warn_and_balance", "warn_and_no_balance", "raise"] = "raise",
        drop_last: bool = False,
    ):
        """
        Initializes a BalancedBatchSampler object.

        Args:
            dataset (Dataset): The dataset to sample from.
            batch_size (int): The size of each batch.
            num_replicas (int): The number of processes participating in distributed training.
            rank (int): The rank of the current process in distributed training.
            device (torch.device): The device to use for the batches.
            mode (str or bool, optional): The mode to use for balancing the batches. Defaults to "atoms".
            shuffle (bool, optional): Whether to shuffle the samples. Defaults to True.
            on_error (Literal["warn_and_balance", "warn_and_no_balance", "raise"], optional): The action to take when an error occurs (i.e., when we have an invalid dataset). Defaults to "raise".
                - "warn_and_balance": Raise a warning and balance the batch by manually loading the data samples and counting the number of nodes (this is slow).
                - "warn_and_no_balance": Raise a warning and do not do any balancing.
                - "raise": Raise an error.
            drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to False.
        """
        self.disabled = False
        self.on_error = on_error

        if mode is False:
            logging.warning(f"Disabled BalancedBatchSampler because {mode=}.")
            self.disabled = True
        elif mode.lower() != "atoms":
            raise ValueError(
                f"Only mode='atoms' or mode=True is supported, got {mode=}."
            )
        elif num_replicas == 1:
            logging.warning(f"Disabled BalancedBatchSampler because {num_replicas=}.")
            self.disabled = True
        if not self.disabled:
            try:
                dataset = _ensure_supported(dataset)
            except ValueError as error:
                if self.on_error == "raise":
                    raise error
                if self.on_error == "warn_and_balance":
                    logging.warning(
                        f"Failed to get data sizes from metadata, loading data to get sizes (THIS IS SLOW). {error}"
                    )
                elif self.on_error == "warn_and_no_balance":
                    logging.warning(
                        f"Failed to get data sizes, falling back to uniform partitioning. {error}"
                    )
                else:
                    raise ValueError(f"Unknown on_error={self.on_error}") from error

        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
            batch_size=batch_size,
            seed=seed,
        )

        super().__init__(sampler, batch_size=batch_size, drop_last=drop_last)
        self.device = device

        logging.info(
            f"Created BalancedBatchSampler with {sampler=}, {batch_size=}, {drop_last=}"
        )

    def _get_natoms(self, batch_idx: list[int]):
        if self.sampler.dataset.metadata_hasattr("natoms"):
            return np.array(
                self.sampler.dataset.get_metadata("natoms", batch_idx)
            ).reshape(-1)
        if self.on_error == "warn_and_balance":
            return np.array([self.sampler.dataset[idx].num_nodes for idx in batch_idx])
        return None

    def set_epoch_and_start_iteration(self, epoch: int, start_iteration: int) -> None:
        if not isinstance(self.sampler, StatefulDistributedSampler):
            if start_iteration != 0:
                raise NotImplementedError(
                    f"{type(self.single_sampler)} does not support resuming from a nonzero step."
                )
            self.sampler.set_epoch(epoch)
        else:
            self.sampler.set_epoch_and_start_iteration(epoch, start_iteration)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

    @staticmethod
    def _dist_enabled():
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    @override
    def __iter__(self):
        if self.disabled or not self._dist_enabled():
            yield from super().__iter__()
            return

        for batch_idx in super().__iter__():
            sizes = self._get_natoms(batch_idx)
            if sizes is None:  # on_error == "warn_and_no_balance" is set
                yield batch_idx
                continue

            idx_sizes = torch.stack(
                [
                    torch.tensor(batch_idx, device=self.device),
                    torch.tensor(sizes, device=self.device),
                ]
            )
            idx_sizes_all = distutils.all_gather(idx_sizes, device=self.device)
            idx_sizes_all = torch.cat(idx_sizes_all, dim=-1).cpu()
            if gp_utils.initialized():
                idx_sizes_all = torch.unique(input=idx_sizes_all, dim=1)
            idx_all = idx_sizes_all[0]
            sizes_all = idx_sizes_all[1]

            local_idx_balanced = _balanced_partition(
                sizes_all.numpy(),
                num_parts=self.sampler.num_replicas,
            )
            # Since DistributedSampler pads the last batch
            # this should always have an entry for each replica.
            yield idx_all[local_idx_balanced[self.sampler.rank]]


@registry.register_logger("new_wandb")
class New_WandBLogger(Logger):
    def __init__(self, config) -> None:
        super().__init__(config)
        (
            self.config["logger"].get("project", None)
            if isinstance(self.config["logger"], dict)
            else None
        )
        (
            self.config["logger"].get("entity", None)
            if isinstance(self.config["logger"], dict)
            else None
        )
        (
            self.config["logger"].get("group", None)
            if isinstance(self.config["logger"], dict)
            else None
        )
        print("master config is :", self.config)
        print("config cmd is :", self.config["cmd"])

        wandb.init(
            config=self.config,
            # id=self.config["cmd"]["timestamp_id"],
            name=self.config["cmd"]["timestamp_id"],
            # dir=self.config["cmd"]["logs_dir"],
            project="SCA",
            # entity="ai4s-sfm", # must set to this, you can use microsoft wandb (ai4s wandb).
            group=self.config["cmd"]["timestamp_id"][:4],
        )

    def watch(self, model, log="all", log_freq: int = 1000) -> None:
        wandb.watch(model, log=log, log_freq=log_freq)

    def log(self, update_dict, step: int, split: str = "") -> None:
        update_dict = super().log(update_dict, step, split)
        wandb.log(update_dict, step=int(step))

    def log_plots(self, plots, caption: str = "") -> None:
        assert isinstance(plots, list)
        plots = [wandb.Image(x, caption=caption) for x in plots]
        wandb.log({"data": plots})

    def log_table(
        self, name: str, cols: list, data: list, step: int | None = None, commit=False
    ) -> None:
        # cols are 1D list of N elements, data must be NxK where the number of cols must match cols
        # see https://docs.wandb.ai/guides/tables
        table = wandb.Table(columns=cols, data=data)
        wandb.log({name: table}, step=step, commit=commit)

    def log_summary(self, summary_dict: dict[str, Any]):
        for k, v in summary_dict.items():
            wandb.run.summary[k] = v

    def mark_preempting(self) -> None:
        wandb.mark_preempting()

    def log_artifact(self, name: str, type: str, file_location: str) -> None:
        art = wandb.Artifact(name=name, type=type)
        art.add_file(file_location)
        art.save()


class Runner(Checkpointable):
    def __init__(self) -> None:
        self.config = None

    def __call__(self, config: dict) -> None:
        with new_trainer_context(config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer
            self.task.setup(self.trainer)
            self.task.run()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        logging.info(
            f'Checkpointing callback is triggered, checkpoint saved to: {self.config["checkpoint"]}, timestamp_id: {self.config["timestamp_id"]}'
        )
        return DelayedSubmission(new_runner, self.config)


def runner_wrapper(config: dict):
    Runner()(config)


def main(
    args: argparse.Namespace | None = None, override_args: list[str] | None = None
):
    """Run the main fairchem program."""
    setup_logging()

    if args is None:
        parser: argparse.ArgumentParser = flags.get_parser()
        parser.add_argument("--nersc", action="store_true", help="Run with NERSC")
        args, override_args = parser.parse_known_args()

    # TODO: rename num_gpus -> num_ranks everywhere
    assert (
        args.num_gpus > 0
    ), "num_gpus is used to determine number ranks, so it must be at least 1"
    if args.num_gpus > 1:
        args.distributed = True
    config = build_config(args, override_args)
    # data_dir = os.path.dirname(os.path.abspath(__file__))  + '/'
    # config["dataset"]['train']['transforms']['element_references']['energy']['file'] = data_dir+config["dataset"]['train']['transforms']['element_references']['energy']['file']
    # config["dataset"]['train']['oc20_ref'] = data_dir + config["dataset"]['train']['oc20_ref']
    if args.timestamp_id is not None and len(args.identifier) == 0:
        args.identifier = args.timestamp_id

    if args.submit:  # Run on cluster
        slurm_add_params = config.get("slurm", None)  # additional slurm arguments
        if args.nersc:
            slurm_add_params["gpus"] = (
                args.num_gpus * args.num_nodes
            )  # total number of gpus, required for NERSC
        configs = create_grid(config, args.sweep_yml) if args.sweep_yml else [config]

        logging.info(f"Submitting {len(configs)} jobs")
        executor = AutoExecutor(folder=args.logdir / "%j", slurm_max_num_timeout=3)
        executor.update_parameters(
            name=args.identifier,
            # mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            # slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(config["optim"]["num_workers"] + 1),
            tasks_per_node=args.num_gpus,
            nodes=args.num_nodes,
            slurm_additional_parameters=slurm_add_params,
            slurm_qos=args.slurm_qos,
            slurm_account=args.slurm_account,
        )
        if not args.nersc:
            executor.update_parameters(
                mem_gb=args.slurm_mem,
                slurm_partition=args.slurm_partition,
            )
        for config in configs:
            config["slurm"] = copy.deepcopy(executor.parameters)
            config["slurm"]["folder"] = str(executor.folder)
        jobs = executor.map_array(Runner(), configs)
        logging.info(f"Submitted jobs: {', '.join([job.job_id for job in jobs])}")
        log_file = save_experiment_log(args, jobs, configs)
        logging.info(f"Experiment log saved to: {log_file}")

    else:  # Run locally on a single node, n-processes
        if args.num_gpus > 1:
            logging.info(f"Running in local mode with {args.num_gpus} ranks")
            # HACK to disable multiprocess dataloading in local mode
            # there is an open issue where LMDB's environment cannot be pickled and used
            # during torch multiprocessing https://github.com/pytorch/examples/issues/526
            if "optim" in config and "num_workers" in config["optim"]:
                config["optim"]["num_workers"] = 0
                logging.info(
                    "WARNING: running in local mode, setting dataloading num_workers to 0, see https://github.com/pytorch/examples/issues/526"
                )

            launch_config = LaunchConfig(
                min_nodes=1,
                max_nodes=1,
                nproc_per_node=args.num_gpus,
                rdzv_backend="c10d",
                max_restarts=0,
            )
            elastic_launch(launch_config, runner_wrapper)(config)
        else:
            logging.info(
                "Running in local mode without elastic launch (single gpu only)"
            )
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["RANK"] = "0"
            os.environ["MASTER_PORT"] = str(get_free_port())
            runner_wrapper(config)


if __name__ == "__main__":
    main()
