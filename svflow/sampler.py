"""
Multi-Trajectory SV-Flow Sampling

Orchestrates N parallel ODE trajectories with SVGD-coupled interaction dynamics.
Uses batched dynamics calls for efficiency: all N trajectories are processed in
a single forward pass through the GVP network.

Core algorithm (simplified — §4.5):
1. Batched forward: all N ligands → single dynamics._forward() call
2. Kinematic decoupling: decompose v_θ into v_int and v_CoM per molecule
3. Predict clean coordinates x̂_0 for CoM computation
4. SVGD repulsive velocity in CoM space (R^3) — isotropic repulsion as baseline
5. Time-annealed scheduling λ(t) — late-onset, parabolic ramp
6. Direct broadcast to atoms (no projection, no orthogonalization)

Key design decisions:
- SVGD operates ONLY on CoM space (R^3): v_int is NEVER modified.
- No tangent plane projection, no orthogonal preservation.
  Ablation experiments proved these constraints DESTROY spatial diversity
  (reducing centroid variance to 7-44% of DrugFlow baseline).
  Pure SVGD repulsion + time annealing is the optimal configuration.
"""

import torch
import numpy as np
from typing import Optional, Union, List

from src.data.molecule_builder import build_molecule
from src.data import data_utils
from src.data.data_utils import TensorDict
from src.analysis.visualization_utils import pocket_to_rdkit

from svflow.kinematics import compute_center_of_mass
from svflow.svgd import compute_svgd_velocity, compute_isotropic_repulsion
from svflow.time_scheduler import TimeAnnealedScheduler


class SVFlowSampler:
    """
    Multi-trajectory SV-Flow sampler wrapping a pretrained DrugFlow model.

    Simplified algorithm: kinematic decoupling → SVGD repulsion (CoM only)
    → time annealing → direct broadcast. No extra geometric constraints.

    Variant configuration (for ablation studies):
      - SV-Flow Core (default):            use_svgd_kernel=True
      - Isotropic Repulsion baseline:      use_svgd_kernel=False

    Usage:
        sampler = SVFlowSampler(model, n_trajectories=10)
        rdmols, rdpockets, info = sampler.sample(pocket_data, timesteps=500)
    """

    def __init__(
        self,
        model,
        n_trajectories: int = 10,
        lambda_max: float = 1.0,
        t_on: float = 0.5,
        d_min: float = 2.0,
        kernel_bandwidth: Optional[float] = None,
        use_svgd_kernel: bool = True,
        verbose: bool = False,
    ):
        self.model = model
        self.N = n_trajectories
        self.lambda_max = lambda_max
        self.d_min = d_min
        self.kernel_bandwidth = kernel_bandwidth
        self.use_svgd_kernel = use_svgd_kernel
        self.verbose = verbose
        self.scheduler = TimeAnnealedScheduler(t_on=t_on, lambda_max=lambda_max)

    @torch.no_grad()
    def sample(
        self,
        pocket_data: dict,
        num_nodes: Union[int, str, None] = None,
        timesteps: Optional[int] = None,
        save_trajectory_kpe: bool = False,
    ):
        """Generate N diverse molecules for a single protein pocket."""
        model = self.model
        device = next(model.parameters()).device

        # Prepare pocket: repeat N times for batched dynamics
        pocket = self._prepare_pocket(pocket_data)
        pocket = data_utils.repeat_items(pocket, self.N)

        T = timesteps if timesteps is not None else model.T_sampling
        delta_t = 1.0 / T

        # Initialize N ligands
        sizes_list = self._sample_sizes(pocket, num_nodes)
        ligands = [self._init_single_ligand(s, pocket) for s in sizes_list]

        n_samples = self.N
        kpe_trajectories = [0.0] * self.N if save_trajectory_kpe else None

        for step in range(T):
            # Time convention: DrugFlow uses t=0→1 (noise→data),
            # SV-Flow scheduler uses t=1→0 (noise→data).
            t_df = step * delta_t                              # DrugFlow: 0→1
            t_sv = 1.0 - t_df                                   # SV-Flow: 1→0
            t_df_array = torch.full((n_samples, 1), fill_value=t_df, device=device)
            t_sv_array = torch.full((n_samples, 1), fill_value=t_sv, device=device)
            lam_t = self.scheduler(t_sv_array)

            # ---- Batched dynamics forward pass (all N ligands at once) ----
            x_cat, h_cat, mask_cat, bonds_cat, bond_types_cat, edge_mask_cat = \
                self._cat_ligands(ligands)

            # Build zero-valued self-conditioning inputs to maintain (s, V) tuple
            # structure required by GVP layers when self_conditioning=True.
            # Zero values ensure no cross-trajectory contamination.
            h_atoms_sc = (
                torch.zeros_like(h_cat),                          # scalar logits
                torch.zeros(x_cat.shape[0], 1, 3, device=device),  # vector velocity
            )
            e_atoms_sc = torch.zeros(
                bond_types_cat.shape[0], model.dynamics.bond_nf, device=device
            ) if bond_types_cat.shape[0] > 0 else torch.zeros(0, model.dynamics.bond_nf, device=device)

            pred_ligand, _ = model.dynamics._forward(
                x_cat, h_cat, mask_cat,
                pocket, t_df_array,  # DrugFlow convention: full t array per trajectory
                bonds_ligand=(bonds_cat, bond_types_cat),
                h_atoms_sc=h_atoms_sc,
                e_atoms_sc=e_atoms_sc,
                h_residues_sc=None,
            )

            # Split batched outputs back per trajectory
            vel = pred_ligand['vel']
            logits_h = pred_ligand['logits_h']
            logits_e = pred_ligand['logits_e']

            vel_list = list(torch.split(vel, sizes_list))
            logits_h_list = list(torch.split(logits_h, sizes_list))
            logits_e_counts = [(s * (s - 1)) // 2 for s in sizes_list]
            logits_e_list = list(torch.split(logits_e, logits_e_counts)) if logits_e.shape[0] > 0 else [logits_e] * self.N

            # Compute per-trajectory quantities
            all_x_hat_0 = []
            all_v_com = []

            for i in range(self.N):
                v_i = vel_list[i]
                x_i = ligands[i]['x']

                x_hat_0 = model.module_x.get_z1_given_zt_and_pred(
                    x_i, v_i, None, t_df_array[i:i+1], ligands[i]['mask']
                )
                all_x_hat_0.append(x_hat_0)
                all_v_com.append(v_i.mean(dim=0))

            # ---- SV-Flow Guidance (simplified: no TP, no OP) ----
            if lam_t.max() > 0:
                com_positions = torch.stack([x.mean(dim=0) for x in all_x_hat_0])

                # (a) Repulsive velocity in CoM space (R^3)
                if self.use_svgd_kernel:
                    delta_v = compute_svgd_velocity(
                        com_positions, d_min=self.d_min, h=self.kernel_bandwidth
                    )
                else:
                    # Isotropic 1/r² distance repulsion baseline
                    delta_v = compute_isotropic_repulsion(
                        com_positions, d_min=self.d_min
                    )

                # (b) Apply time-annealed guidance directly — no projections
                guidance = lam_t * delta_v  # (N, 3)
            else:
                guidance = torch.zeros(self.N, 3, device=device)

            # ---- Apply ODE step to each trajectory ----
            for i in range(self.N):
                t_next = t_df_array[i:i+1] + delta_t
                t_cur = t_df_array[i:i+1]

                # Coordinate update: base velocity + guidance broadcast to all atoms
                vel_guided = vel_list[i] + guidance[i:i+1]
                ligands[i]['x'] = model.module_x.sample_zt_given_zs(
                    ligands[i]['x'], vel_guided, t_cur, t_next, ligands[i]['mask']
                )

                # Atom/bond type update (unmodified base model logits)
                ligands[i]['h'] = model.module_h.sample_zt_given_zs(
                    ligands[i]['h'], logits_h_list[i], t_cur, t_next, ligands[i]['mask']
                )
                ligands[i]['e'] = model.module_e.sample_zt_given_zs(
                    ligands[i]['e'], logits_e_list[i], t_cur, t_next, ligands[i]['edge_mask']
                )

                if save_trajectory_kpe:
                    kpe_trajectories[i] += (all_v_com[i] ** 2).sum().item() * delta_t

        # Build output
        rdmols = self._build_molecules(ligands)
        rdpockets = pocket_to_rdkit(
            pocket, model.pocket_representation,
            model.atom_encoder, model.atom_decoder,
            model.aa_decoder, model.residue_decoder,
            model.aa_atom_index,
        )

        info = {
            'n_trajectories': self.N, 'timesteps': T, 'ligand_sizes': sizes_list,
        }
        if save_trajectory_kpe:
            info['kpe_per_trajectory'] = kpe_trajectories

        return rdmols, rdpockets, info

    # ------------------------------------------------------------------
    # Batching helpers
    # ------------------------------------------------------------------

    def _cat_ligands(self, ligands: List[TensorDict]):
        """Concatenate N ligands into batched tensors for dynamics._forward().

        Returns:
            x_cat: all atom coords, shape (sum(sizes), 3)
            h_cat: all atom features, shape (sum(sizes), atom_nf)
            mask_cat: batch mask assigning atoms to trajectories, shape (sum(sizes),)
            bonds_cat: bond indices (offset-adjusted), shape (2, sum(n_bonds))
            bond_types_cat: bond type one-hots, shape (sum(n_bonds), bond_nf)
            edge_mask_cat: batch mask for bonds, shape (sum(n_bonds),)
        """
        sizes = [l['x'].shape[0] for l in ligands]
        offsets = [0] + list(torch.tensor(sizes[:-1]).cumsum(dim=0).tolist())

        x_cat = torch.cat([l['x'] for l in ligands], dim=0)
        h_cat = torch.cat([l['h'] for l in ligands], dim=0)
        mask_cat = torch.cat([
            torch.full((s,), i, dtype=torch.long, device=x_cat.device)
            for i, s in enumerate(sizes)
        ])

        # Offset bond indices
        bonds_list = []
        bond_types_list = []
        edge_mask_list = []
        for i, (l, off) in enumerate(zip(ligands, offsets)):
            bonds_i = l['bonds'] + off
            bonds_list.append(bonds_i)
            bond_types_list.append(l['e'])
            edge_mask_list.append(l['edge_mask'] if l['edge_mask'].numel() > 0 else
                                  torch.arange(bonds_i.shape[1], device=x_cat.device) * 0 + i)

        bonds_cat = torch.cat(bonds_list, dim=1) if bonds_list else torch.zeros(2, 0, dtype=torch.long, device=x_cat.device)
        bond_types_cat = torch.cat(bond_types_list, dim=0) if bond_types_list else torch.zeros(0, bond_types_list[0].shape[-1] if bond_types_list else 5, device=x_cat.device)
        edge_mask_cat = torch.cat(edge_mask_list, dim=0) if edge_mask_list else torch.zeros(0, dtype=torch.long, device=x_cat.device)

        return x_cat, h_cat, mask_cat, bonds_cat, bond_types_cat, edge_mask_cat

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_pocket(self, pocket_data: dict):
        from src.data.data_utils import Residues
        if isinstance(pocket_data.get('pocket'), Residues):
            return pocket_data['pocket']
        return Residues(**pocket_data['pocket'])

    def _sample_sizes(self, pocket, num_nodes_spec) -> List[int]:
        model = self.model
        if isinstance(num_nodes_spec, str) and num_nodes_spec.startswith("uniform"):
            parts = num_nodes_spec.split("_")
            left, right = int(parts[1]), int(parts[2])
            return [torch.randint(left, right + 1, (1,)).item() for _ in range(self.N)]
        elif isinstance(num_nodes_spec, int):
            return [num_nodes_spec] * self.N
        else:
            sizes = []
            for _ in range(self.N):
                n = model.size_distribution.sample_conditional(
                    n1=None, n2=pocket['size'][:1]
                )[0].item()
                sizes.append(max(n, 2))
            return sizes

    def _init_single_ligand(self, num_nodes: int, pocket) -> TensorDict:
        model = self.model
        device = pocket['x'].device

        lig_mask = torch.zeros(num_nodes, dtype=torch.long, device=device)
        lig_bonds = torch.stack(torch.where(torch.triu(
            lig_mask[:, None] == lig_mask[None, :], diagonal=1)), dim=0)
        lig_edge_mask = (
            lig_mask[lig_bonds[0]]
            if lig_bonds.numel() > 0
            else torch.zeros(0, dtype=torch.long, device=device)
        )

        pocket_com = (
            pocket['x'].mean(dim=0, keepdim=True)
            if pocket['x'].numel() > 0
            else torch.zeros(1, 3, device=device)
        )

        z0_x = model.module_x.sample_z0(pocket_com, lig_mask)
        z0_h = model.module_h.sample_z0(lig_mask)
        z0_e = model.module_e.sample_z0(lig_edge_mask)

        return TensorDict(**{
            'x': z0_x, 'h': z0_h, 'e': z0_e, 'mask': lig_mask,
            'bonds': lig_bonds, 'edge_mask': lig_edge_mask,
        })

    def _build_molecules(self, ligands) -> list:
        model = self.model
        rdmols = []
        for lig in ligands:
            x = lig['x'].detach().cpu()
            h = lig['h'].argmax(dim=-1).detach().cpu()
            e = lig['e'].argmax(dim=-1).detach().cpu()
            mol = build_molecule(
                x, h,
                bonds=lig['bonds'].detach().cpu(),
                bond_types=e,
                atom_decoder=model.atom_decoder,
                bond_decoder=model.bond_decoder,
            )
            rdmols.append(mol)
        return rdmols
