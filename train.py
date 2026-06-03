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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_rotation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def print_memory(tag):
    print(f"[{tag}] allocated={torch.cuda.memory_allocated()/1e9:.3f}GB reserved={torch.cuda.memory_reserved()/1e9:.3f}GB max_reserved={torch.cuda.max_memory_reserved()/1e9:.3f}GB")

# Instrumentación de debug, activada por variables de entorno (0/ausente = off):
#   DEBUG_MEM=N    -> imprime memoria cada N iters, en cada punto del step (post_render/
#                     post_backward/post_step) con el PICO por iteración reseteado al inicio.
#   DEBUG_NOISE=N  -> imprime estadísticas del ruido posicional cada N aplicaciones de ruido.
DEBUG_MEM = int(os.environ.get("DEBUG_MEM", "0"))
DEBUG_NOISE = int(os.environ.get("DEBUG_NOISE", "0"))

def mem_probe(tag, iteration, npts=None):
    """Una línea compacta con alloc/reserved actuales, el pico de la iteración y la
    memoria LIBRE real del device (delata procesos zombie ocupando VRAM)."""
    if not DEBUG_MEM or iteration % DEBUG_MEM != 0:
        return
    a  = torch.cuda.memory_allocated() / 1e9
    r  = torch.cuda.memory_reserved() / 1e9
    pa = torch.cuda.max_memory_allocated() / 1e9
    pr = torch.cuda.max_memory_reserved() / 1e9
    free, total = torch.cuda.mem_get_info()
    extra = f" npts={npts}" if npts is not None else ""
    print(f"[MEM it{iteration} {tag}] alloc={a:.2f} reserved={r:.2f} "
          f"peak_alloc={pa:.2f} peak_reserved={pr:.2f} "
          f"dev_free={free/1e9:.2f}/{total/1e9:.2f}GB{extra}")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        # Pico de memoria POR step: lo reseteamos aquí para que peak_alloc/peak_reserved
        # midan el máximo dentro de esta iteración (delata el spike del binning del rasterizer).
        if DEBUG_MEM and iteration % DEBUG_MEM == 0:
            torch.cuda.reset_peak_memory_stats()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        mem_probe("post_render", iteration, npts=gaussians.get_xyz.shape[0])
                
        # 🔍 Debug beta (cada 100 iteraciones)
        if iteration % 5000 == 0:
            with torch.no_grad():
                beta = gaussians.get_beta
                print(
                    f"[Iter {iteration}] "
                    f"beta mean={beta.mean().item():.4f}, "
                    f"min={beta.min().item():.4f}, "
                    f"max={beta.max().item():.4f}"
                )

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]        
        if torch.isfinite(normal_error).all():
            normal_loss = lambda_normal * normal_error.mean()
        else:
            normal_loss = 0.0
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss

        # Regularizers (matching official Beta Splatting): opacity_reg + scale_reg L1.
        # Active only during densification window, as in beta-splatting/train.py.
        if opt.densify_from_iter < iteration < opt.densify_until_iter:
            total_loss = total_loss + opt.opacity_reg * gaussians.get_opacity.abs().mean()
            total_loss = total_loss + opt.scale_reg * gaussians.get_scaling.abs().mean()

        total_loss.backward()
        mem_probe("post_backward", iteration, npts=gaussians.get_xyz.shape[0])

        # ✅ DEBUG gradiente de beta (DESPUÉS del backward)
        if iteration % 10000 == 0 and gaussians._beta.grad is not None:
            print(
                "grad beta mean:",
                gaussians._beta.grad.mean().item()
            )

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # Prune definitivo de NaN antes de guardar: evita zonas negras al renderizar.
                gaussians.prune_nan_splats(iteration=iteration)
                scene.save(iteration)


            # Acumular señal de error de reconstrucción por splat (norma del gradiente
            # de viewspace). La consume el muestreo MCMC sesgado por error en
            # relocate_gs/add_new_gs. Se acumula en cada iteración del intervalo;
            # densification_postfix la resetea al añadir splats.
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            # Densification (MCMC, alineado con Beta Splatting oficial).
            # Solo relocate + add_new (sin densify_and_prune 2DGS). Ruido posicional
            # se aplica únicamente en los pasos de densify para evitar la divergencia
            # numérica que se observó en Run 2 (NaN tras iter 30000).
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # Defensa: sanear NaN/Inf que puedan haber entrado por gradientes
                    # explosivos o por desplazamientos extremos del ruido posicional.
                    gaussians.sanitize_parameters(iteration=iteration)
                    dead_mask = (gaussians.get_opacity <= opt.opacity_cull).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask, error_weight=opt.mcmc_error_weight)
                    gaussians.add_new_gs(cap_max=opt.cap_max, error_weight=opt.mcmc_error_weight, jitter_scale=opt.mcmc_jitter_scale)

                    # Ruido posicional MCMC. Dos modos:
                    #  - isotrópico (default): randn esférico, ignora la forma del splat.
                    #  - híbrido covarianza (opt.cov_noise): anisotropía en el PLANO del
                    #    surfel (las 2 escalas) + isotrópico en la NORMAL. No usamos
                    #    build_scaling_rotation (accede a s[:,2] → IndexError con scales 2D);
                    #    construimos R con build_rotation y aplicamos un std local de 3 ejes.
                    #    OJO magnitudes: usamos pesos de anisotropía NORMALIZADOS a media 1
                    #    (no s² crudo, que es ~1e-4 y descuadraría noise_lr congelando el
                    #    plano); así noise_lr sigue calibrado. Ver docs/ruido_isotropico_*.
                    with torch.no_grad():
                        noise_exp = float(getattr(opt, "noise_opacity_exponent", 100.0))
                        noise_lr = float(getattr(opt, "noise_lr", 5e5))
                        base_noise = torch.randn_like(gaussians._xyz)
                        noise_mult = torch.pow(1.0 - gaussians.get_opacity, noise_exp)
                        if bool(getattr(opt, "cov_noise", False)):
                            s = gaussians.get_scaling                                  # (N,2) ejes del plano
                            w = s / s.mean(dim=1, keepdim=True).clamp_min(1e-8)        # anisotropía, media 1
                            pad = float(getattr(opt, "cov_noise_normal", 1.0))         # 0 = confinado al plano
                            local_std = torch.cat(
                                [w, torch.full_like(s[:, :1], pad)], dim=1)            # (N,3): [u, v, normal]
                            R = build_rotation(gaussians.get_rotation)                 # (N,3,3), col 2 = normal
                            base_noise = torch.bmm(
                                R, (local_std * base_noise).unsqueeze(-1)).squeeze(-1) # moldea y rota al mundo
                        noise = base_noise * noise_mult * noise_lr * float(xyz_lr)

                        # Debug del ruido: ¿qué splats se mueven y cuánto? Con noise_exp=100
                        # solo los de opacidad baja deberían moverse; el desplazamiento debe
                        # ser pequeño frente a la escala de escena (spatial_lr_scale).
                        if DEBUG_NOISE and (iteration // opt.densification_interval) % DEBUG_NOISE == 0:
                            op = gaussians.get_opacity.squeeze(-1)
                            disp = noise.norm(dim=1)                         # |Δxyz| por splat
                            mult = noise_mult.squeeze(-1)
                            active = mult > 1e-3                            # splats realmente perturbados
                            print(f"[NOISE it{iteration}] N={op.numel()} "
                                  f"op(mean={op.mean().item():.3f} min={op.min().item():.3f}) "
                                  f"mult(mean={mult.mean().item():.2e} max={mult.max().item():.2e}) "
                                  f"activos={int(active.sum().item())} ({100*active.float().mean().item():.2f}%) "
                                  f"|disp|(mean={disp.mean().item():.2e} med={disp.median().item():.2e} "
                                  f"max={disp.max().item():.2e}) "
                                  f"xyz_lr={float(xyz_lr):.2e} noise_lr={noise_lr:.1e} extent={gaussians.spatial_lr_scale:.3f}")
                            if bool(getattr(opt, "cov_noise", False)):
                                print(f"[NOISE it{iteration}] cov: w(mean={w.mean().item():.3f} "
                                      f"min={w.min().item():.3f} max={w.max().item():.3f}) "
                                      f"pad(normal)={pad:.2f}")

                        gaussians._xyz.add_(noise)

                # Reset de opacidades periódico (default deshabilitado: interval=1e9)
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print_memory("antes_reset")
                    gaussians.reset_opacity()
                    print_memory("después_reset")
            else:
                # Post-densify: sanear NaN/Inf cada densification_interval pasos para que
                # los últimos miles de iters no acumulen splats degenerados (zonas negras).
                if iteration % opt.densification_interval == 0:
                    gaussians.sanitize_parameters(iteration=iteration)

            # Optimizer step
            if iteration < opt.iterations:
                # NaN-safe: limpia gradientes con NaN/Inf antes de step. Evita que un
                # único batch con gradiente patológico contamine los parámetros para
                # siempre. Cubre TODAS las iters (no solo las de densificación).
                for p in gaussians.optimizer.param_groups:
                    for tensor in p['params']:
                        if tensor.grad is not None:
                            torch.nan_to_num_(tensor.grad, nan=0.0, posinf=0.0, neginf=0.0)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            mem_probe("post_step", iteration, npts=gaussians.get_xyz.shape[0])

            # ✅ Clamp suave del parámetro b (no del beta)
            with torch.no_grad():
                gaussians._beta.data.clamp_(min=-4.0, max=2.0)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)   
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
