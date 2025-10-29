
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
from pathlib import Path

import numpy as np
import torch
import torch.distributed

from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

if TYPE_CHECKING:
    import argparse
from fairchem.core.common.registry import registry

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


import ase
from fairchem.core.datasets.base_dataset import BaseDataset

# set the cuda home
from fairchem.core.datasets.oc22_lmdb_dataset import OC22LmdbDataset
from fairchem.core.modules.transforms import DataTransforms
from torch.nn.parallel.distributed import DistributedDataParallel
from torch_geometric.data import Data


@registry.register_dataset("spice_xyz")
class SPICExyzDataset(torch.utils.data.Dataset):
    def __init__(self, config, path = None,transform=None,**args) -> None:
        super().__init__()
        
        self.config = config
        if path is None:
            self.paths = []

            if "src" in self.config:
                if isinstance(config["src"], str):
                    self.paths = [Path(self.config["src"])]
                else:
                    self.paths = tuple(Path(path) for path in sorted(config["src"]))

            assert (
                len(self.paths) == 1
            ), f"{type(self)} does not support a list of src paths."
            self.path = self.paths[0]
            self.transforms = DataTransforms(self.config.get("transforms", {}))

        else:
            self.path = path
        self.transforms = DataTransforms({})
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
                "data_name":"spice-maceoff233",
                "idx": idx,

                
                "atomic_numbers": torch.from_numpy(atomic_numbers),
                "pos": torch.from_numpy(pos).float(),
                "num_atoms": torch.tensor([len(atomic_numbers)], dtype=torch.int),
                "pbc": torch.zeros(3),
                "cell": torch.zeros(1, 3, 3),
                "charge": torch.tensor([0], dtype=torch.int).reshape(-1),
                "multiplicity": torch.tensor([0], dtype=torch.int).reshape(-1),
                
                "energy": torch.from_numpy(energy.reshape(1)).float(),
                "energy_per_atom":torch.from_numpy(energy.reshape(1)).float() / len(atomic_numbers),
                "forces": torch.from_numpy(forces).float(),


                "subset_name": self.subset_name2id[subset_name],
                "forces_subset_name": torch.zeros(len(atomic_numbers))
                + self.subset_name2id[subset_name],
                
            }
        )
        
        return self.transforms(out)
    
    def __len__(self) -> int:
        return self.num_samples

@registry.register_dataset("TOY_xyz")
class TOYxyzDataset():
    def __init__(self, config, path = None,transform=None,**args) -> None:
        super().__init__()
        
       
    def __getitem__(self, idx):

        N = 3000
        rho = 0.8
        volume = N / rho
        box_length = volume ** (1/3)

        # Generate N random 3D positions in the box
        pos = torch.rand((N, 3)).numpy()*box_length
        atomic_numbers = torch.randint(0,10,(N,)).numpy()
        forces = torch.randn((N, 3)).numpy()
        energy = torch.randn(1).numpy()
        

        out = Data(
            **{
                "data_name":"spice-maceoff233",
                "idx": idx,

                
                "atomic_numbers": torch.from_numpy(atomic_numbers),
                "pos": torch.from_numpy(pos).float(),
                "natoms": len(atomic_numbers),
                "pbc": torch.zeros(3),
                "cell": torch.zeros(1, 3, 3),
                "charge": torch.tensor([0], dtype=torch.int).reshape(-1),
                "multiplicity": torch.tensor([0], dtype=torch.int).reshape(-1),
                
                "energy": torch.from_numpy(energy.reshape(1)).float(),
                "forces": torch.from_numpy(forces).float(),

            }
        )
        
        return out
    
    def __len__(self,):
        return 6400
