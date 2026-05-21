#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import math
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._beta = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        # ✅ DBS: contador de opacidad baja sostenida
        self.low_opacity_counter = torch.empty(0)
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()        

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)    
    
    @property
    def get_beta(self):
        b = self._beta.clamp(min=-6.0, max=2.0)
        return (4.0 * torch.exp(b)).contiguous()

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))       
        betas = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        self._beta = nn.Parameter(betas.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.low_opacity_counter = torch.zeros((self.get_xyz.shape[0],), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.beta_densify_threshold = getattr(training_args, "beta_densify_threshold", 0.0)
        self.beta_densify_mode = getattr(training_args, "beta_densify_mode", "split_wide")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._beta], 'lr': training_args.beta_lr, "name": "beta"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        l.append('beta')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        beta = self._beta.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, beta, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        # ✅ beta: inicializado de forma consistente con el resto
        beta = np.ones((xyz.shape[0], 1), dtype=np.float32)
        self._beta = nn.Parameter(torch.tensor(beta, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._beta = optimizable_tensors["beta"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.low_opacity_counter = self.low_opacity_counter[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_beta, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "beta": new_beta,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.low_opacity_counter = torch.zeros((self.get_xyz.shape[0],), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # ---------- FILTRO OPCIONAL POR BETA (insertar aquí) ----------
        # Requiere que hayas añadido `self.beta_densify_threshold` (por ejemplo en arguments.py)
        # y que la hayas inicializado en la instancia (p.ej. en training_setup).
        if hasattr(self, "beta_densify_threshold") and self.beta_densify_threshold > 0.0:
            beta_vals = self.get_beta.squeeze()
            if beta_vals.shape[0] != n_init_points:
                beta_vals = beta_vals[:n_init_points]
            mode = getattr(self, "beta_densify_mode", "split_wide")
            if mode == "split_wide":
                selected_pts_mask = selected_pts_mask & (beta_vals <= self.beta_densify_threshold)
            else:
                selected_pts_mask = selected_pts_mask & (beta_vals >= self.beta_densify_threshold)
        # safety: si no hay puntos seleccionados, salir
        if selected_pts_mask.sum() == 0:
            return
        # ---------- fin filtro por beta ----------

        # ===============================
        # ✅ Deterministic DBS split
        # ===============================

        xyz = self.get_xyz[selected_pts_mask]
        scales = self.get_scaling[selected_pts_mask]
        rots = build_rotation(self._rotation[selected_pts_mask])

        # elegir eje dominante (mayor escala)
        mask = (scales[:, 0] > scales[:, 1]).float().unsqueeze(1)

        # eje principal en espacio local
        v1_local = torch.cat([
            mask,                  # eje x si s_x > s_y
            1 - mask,              # eje y si no
            torch.zeros_like(mask) # sin componente normal
        ], dim=1)

        # llevar a espacio mundo
        v1_world = torch.bmm(rots, v1_local.unsqueeze(-1)).squeeze(-1)

        # magnitud del split (proporcional al tamaño)
        delta = 0.5 * torch.max(scales, dim=1).values.unsqueeze(1)

        # crear dos hijos
        new_xyz = torch.cat([
            xyz + delta * v1_world,
            xyz - delta * v1_world
        ], dim=0)
        
        new_scaling = self.scaling_inverse_activation(
            (self.get_scaling[selected_pts_mask] / (0.8 * N)).repeat(N, 1)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)        
        # ✅ DBS-style split: alpha is evenly divided
        alpha = self.get_opacity[selected_pts_mask]
        alpha_new = alpha / N
        new_opacity = self.inverse_opacity_activation(alpha_new).repeat(N, 1)

        new_beta = (self._beta[selected_pts_mask] - math.log(N)).repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_beta, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]

        # ✅ DBS-style clone: preserve transmittance
        alpha = self.get_opacity[selected_pts_mask]      # (0,1)
        K = 1  # clone crea 1 copia adicional por punto
        alpha_new = alpha / (K + 1)

        new_opacities = self.inverse_opacity_activation(alpha_new)

        new_beta = self._beta[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_beta, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        '''
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        '''
        # ===============================
        # ✅ DBS-style pruning (stable)
        # ===============================

        alpha = self.get_opacity.squeeze()

        # ---- (A) Opacidad baja sostenida ----
        low_alpha = alpha < min_opacity

        # actualizar contador
        self.low_opacity_counter[low_alpha] += 1
        self.low_opacity_counter[~low_alpha] = 0

        N_sustain = 50  # puedes ajustar (30–100)
        prune_alpha_mask = self.low_opacity_counter > N_sustain


        # ---- (B) Área tangencial excesiva ----
        scales = self.get_scaling
        area = scales[:, 0] * scales[:, 1]

        max_area = (0.1 * extent) ** 2
        prune_area_mask = area > max_area


        # ---- combinar criterios ----
        prune_mask = prune_alpha_mask | prune_area_mask


        # ---- mantener condición pantalla ----
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = prune_mask | big_points_vs


        # ---- aplicar pruning ----
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def relocate_gs(self, opacity_threshold=0.005, min_opacity_iters=100):
        """
        Relocaliza gaussians con opacidad baja sostenida.
        Solo actúa sobre aquellos que han estado por debajo del umbral por min_opacity_iters.
        """
        cur_opacity = self.get_opacity.squeeze()
        
        # marcar los que están por debajo del umbral en esta iteración
        low_mask = (cur_opacity <= opacity_threshold)
        
        # incrementar contador
        self.low_opacity_counter[low_mask] += 1
        self.low_opacity_counter[~low_mask] = 0  # reset si se recuperan
        
        # identificar "muertos" sostenidos
        dead_mask = (self.low_opacity_counter >= min_opacity_iters)
        n_dead = dead_mask.sum().item()
        
        if n_dead == 0:
            return
        
        print(f"[Relocate] {n_dead} dead gaussians detected, relocating...")
        
        # índices muertos y vivos
        dst_idx = dead_mask.nonzero(as_tuple=True)[0]  # ✅ AQUÍ se define dst_idx
        alive_mask = ~dead_mask
        alive_idx = alive_mask.nonzero(as_tuple=True)[0]
        
        if alive_idx.shape[0] == 0:
            print("[Relocate] No alive gaussians to sample from, skipping.")
            return
        
        # seleccionar fuentes con pesos por opacidad
        alive_op = cur_opacity[alive_idx]
        weights = alive_op.clamp(min=1e-6)
        weights = weights / weights.sum()
        
        # samplear fuentes (con reemplazo si hay más muertos que vivos)
        src_idx = alive_idx[torch.multinomial(weights, n_dead, replacement=True)]
        
        # aplicar jitter a las posiciones
        with torch.no_grad():
            avg_scale = self.get_scaling.mean().mean().clamp(min=1e-6)
            jitter_scale = 0.05 * avg_scale
            jitter = torch.randn((n_dead, self.get_xyz.shape[1]), device=self._xyz.device) * jitter_scale
            
            # copiar xyz + jitter
            self._xyz[dst_idx] = self._xyz[src_idx] + jitter
            
            # copiar features
            self._features_dc[dst_idx] = self._features_dc[src_idx].clone()
            self._features_rest[dst_idx] = self._features_rest[src_idx].clone()
            
            # copiar scaling/rotation
            self._scaling[dst_idx] = self._scaling[src_idx].clone()
            self._rotation[dst_idx] = self._rotation[src_idx].clone()
            
            # opacidad inicial: reducida para evitar saturación
            src_alpha = self.get_opacity[src_idx]
            new_alpha = (src_alpha / 1.5).clamp(min=1e-3, max=0.95)  # ✅ rango ajustado
            self._opacity[dst_idx] = self.inverse_opacity_activation(new_alpha)
            
            # beta: copia de la fuente
            self._beta[dst_idx] = self._beta[src_idx].clone()
            
            # reset counters / stats
            self.low_opacity_counter[dst_idx] = 0
            self.xyz_gradient_accum[dst_idx] = 0.0
            self.denom[dst_idx] = 0.0
            self.max_radii2D[dst_idx] = 0.0

    def add_new_gs(self, cap_max):
        """
        Añade nuevos splats hasta alcanzar cap_max (o hasta un crecimiento razonable).
        Estrategia:
        - target = min(cap_max, int(1.05 * current_num_points))
        - sample fuentes con probabilidad proporcional a opacidad
        - crear nuevos tensores y llamar a densification_postfix para concatenarlos
        """
        cur = self.get_xyz.shape[0]
        
        # target growth cap: 5% sobre current por paso, hasta cap_max
        target = min(int(cap_max), int(math.ceil(1.05 * cur)))
        k = target - cur
        
        if k <= 0:
            return
        
        # preferir fuentes con opacidad alta
        op = self.get_opacity.squeeze()
        weights = op.clamp(min=1e-6)
        weights = weights / weights.sum()
        
        src_idx = torch.multinomial(weights, k, replacement=True)
        
        # construir nuevos tensores a partir de las fuentes (con jitter pequeño)
        with torch.no_grad():
            avg_scale = self.get_scaling.mean().mean().clamp(min=1e-6)
            jitter_scale = 0.05 * avg_scale
            jitter = torch.randn((k, self.get_xyz.shape[1]), device=self._xyz.device) * jitter_scale
            
            new_xyz = (self._xyz[src_idx] + jitter).detach().clone()
            new_features_dc = self._features_dc[src_idx].detach().clone()
            new_features_rest = self._features_rest[src_idx].detach().clone()
            new_scaling = self._scaling[src_idx].detach().clone()
            new_rotation = self._rotation[src_idx].detach().clone()
            
            # Opacidades: dividir para evitar saturación
            src_alpha = self.get_opacity[src_idx]
            new_alpha = (src_alpha / 1.5).clamp(min=1e-3, max=0.95)  # ✅ mismo rango que relocate
            new_opacity = self.inverse_opacity_activation(new_alpha)
            
            # Beta: copiar de la fuente
            new_beta = self._beta[src_idx].detach().clone()
            
            # ✅ usar la función existente para concatenar y actualizar optimizador
            self.densification_postfix(
                new_xyz, 
                new_features_dc, 
                new_features_rest, 
                new_opacity, 
                new_beta,  # asegúrate que densification_postfix acepte beta
                new_scaling, 
                new_rotation
            )