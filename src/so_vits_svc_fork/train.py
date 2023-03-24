from __future__ import annotations

import multiprocessing
import os
import time
from logging import getLogger
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import so_vits_svc_fork.modules.commons as commons

from . import utils
from .data_utils import (
    TextAudioCollate,
    TextAudioSpeakerLoader,
    WorstPerformingTextAudioSpeakerBatchLoader,
)
from .models import Discriminator, SynthesizerTrn
from .modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from .modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'


LOG = getLogger(__name__)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose = False
torch._dynamo.config.suppress_errors = False

global_step = 0
start_time = time.time()

# Disable all debug apis, These should be enabled if a change is made
torch.set_anomaly_enabled(True)


def train(config_path: Path | str, model_path: Path | str):
    """Assume Single Node Multi GPUs Training Only"""
    config_path = Path(config_path)
    model_path = Path(model_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    utils.ensure_pretrained_model(model_path)
    hps = utils.get_hparams(config_path, model_path)

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = hps.train.port

    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )

    # compiled_run = torch.compile(run)
    run(rank=0, n_gpus=n_gpus, hps=hps)


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        LOG.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=Path(hps.model_dir) / "eval")

    # for pytorch on win, backend use gloo
    dist.init_process_group(
        backend="gloo" if os.name == "nt" else "nccl",
        init_method="env://",
        world_size=n_gpus,
        rank=rank,
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)
    collate_fn = TextAudioCollate()
    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps)
    # train_dataset = PreloadedTextAudioSpeakerLoader(hps.data.training_files, hps)
    num_workers = 5 if multiprocessing.cpu_count() > 4 else multiprocessing.cpu_count()
    train_loader = DataLoader(
        train_dataset,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        batch_size=hps.train.batch_size,
        collate_fn=collate_fn,
    )
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=1,
            shuffle=False,
            batch_size=1,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(rank)
    net_d = Discriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = torch.compile(DDP(net_g, device_ids=[rank]))  # , find_unused_parameters=True)
    # net_d = torch.compile(DDP(net_d, device_ids=[rank]))

    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

    # net_d =  DDP(net_d, device_ids=[rank])

    # net_g = net_g
    # net_d = net_d

    skip_optimizer = False
    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
            net_g,
            optim_g,
            skip_optimizer,
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
            net_d,
            optim_d,
            skip_optimizer,
        )
        epoch_str = max(epoch_str, 1)
        global_step = (epoch_str - 1) * len(train_loader)
    except Exception as e:
        LOG.exception(e)
        LOG.info("No checkpoint found, start from scratch")
        epoch_str = 1
        global_step = 0
    if skip_optimizer:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    LOG.info("Start training")

    counter = 0
    build_worst_loader = False
    use_worst_loader = False
    for epoch in trange(epoch_str, hps.train.epochs + 1):
        # if counter == 9:
        #     build_worst_loader = True
        # elif counter == 10:
        #     build_worst_loader = False
        #     use_worst_loader = True
        #     counter = 0
        # else:
        #     build_worst_loader = False
        #     use_worst_loader = False
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, eval_loader],
                [writer, writer_eval],
                build_worst_loader=build_worst_loader,
                use_worst_loader=use_worst_loader,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
            )
        scheduler_g.step()
        scheduler_d.step()
        counter += 1


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    schedulers,
    scaler,
    loaders,
    writers,
    use_worst_loader=False,
    build_worst_loader=False,
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    # train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    ###
    #   FREEZING LOGIC
    ###

    # Test Freeze lower discrim layers
    # for discriminator in net_d.module.discriminators:
    #    discriminator.convs.requires_grad_(False)

    # Test Alternate Discrim layer training
    # alternate_bool = (epoch % 2 == 0)
    # for discriminator in net_d.module.discriminators:
    #    discriminator.convs.requires_grad_(alternate_bool)
    #    discriminator.conv_post.requires_grad_(not alternate_bool)

    # Test Alternate Discrim layer training - improved stability verified
    # alternate_bool = (epoch % 2 == 0)
    # for discriminator in net_d.module.discriminators:
    #    if alternate_bool:
    #        for i, layer in enumerate(discriminator.convs):
    #            layer.requires_grad_((epoch//2) % len(discriminator.convs) == i)
    #    #discriminator.conv_post.requires_grad_(not alternate_bool)
    #    discriminator.conv_post.requires_grad_(True)

    # Freeze everything but the speaker embedding
    # fine_tune_embedding_freeze(net_d=net_d, net_g=net_g)
    net_g.requires_grad_(True)
    net_g.module.requires_grad_(True)
    net_d.requires_grad_(True)
    net_d.module.requires_grad_(True)

    if (use_worst_loader or build_worst_loader) and not (
        "worst_loader" in globals() or "worst_loader" in locals()
    ):
        worst_loader = WorstPerformingTextAudioSpeakerBatchLoader()
    if use_worst_loader:
        worst_loader.prepare()
        train_loader = worst_loader
    ###
    #   /END FREEZING LOGIC
    ###
    with torch.profiler.profile(
        # activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        activities=[],
        schedule=torch.profiler.schedule(wait=2, warmup=50, active=1, repeat=1),
        # record_shapes=True,
        # profile_memory=True,
        # with_stack=True,
        # with_modules=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("test"),
    ) as prof:
        for batch_idx, items in enumerate(train_loader):
            # This worked REALLY well to restart the discriminator by running for 1 iteration
            # Improved training convergence speed
            # if (batch_idx % 3 == 0):
            #    net_g.requires_grad_(True)
            #    net_g.module.requires_grad_(True)
            # else:
            #     net_g.requires_grad_(False)
            #     net_g.module.requires_grad_(False)

            c, f0, spec, y, spk, lengths, uv = items
            g = spk.cuda(rank, non_blocking=True)
            spec, y = spec.cuda(rank, non_blocking=True), y.cuda(
                rank, non_blocking=True
            )
            c = c.cuda(rank, non_blocking=True)
            f0 = f0.cuda(rank, non_blocking=True)
            uv = uv.cuda(rank, non_blocking=True)
            lengths = lengths.cuda(rank, non_blocking=True)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            with autocast(enabled=hps.train.fp16_run):
                (
                    y_hat,
                    ids_slice,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                    pred_lf0,
                    norm_lf0,
                    lf0,
                ) = net_g(c, f0, uv, spec, g=g, c_lengths=lengths, spec_lengths=lengths)

                y_mel = commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
                y = commons.slice_segments(
                    y, ids_slice * hps.data.hop_length, hps.train.segment_size
                )  # slice

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

                # if batch_idx == 50:
                #     print("REAL AVG")
                #     print(list(map(lambda x: torch.mean(x), y_d_hat_r)))
                #     print("REAL MAX")
                #     print(list(map(lambda x: torch.max(x), y_d_hat_r)))
                #     print("REAL MIN")
                #     print(list(map(lambda x: torch.min(x), y_d_hat_r)))
                #     print("GENERATED AVG")
                #     print(list(map(lambda x: torch.mean(x), y_d_hat_g)))
                #     print("GENERATED MAX")
                #     print(list(map(lambda x: torch.max(x), y_d_hat_g)))
                #     print("GENERATED MIN")
                #     print(list(map(lambda x: torch.min(x), y_d_hat_g)))

                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc

            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            scaler.step(optim_d)

            with autocast(enabled=hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_lf0 = F.mse_loss(pred_lf0, lf0)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            scaler.step(optim_g)
            scaler.update()

            if build_worst_loader and not use_worst_loader:
                worst_loader.add_batch(batch=items, loss_tensor=loss_gen_all)

            if rank == 0:
                if global_step % hps.train.log_interval == 0:
                    lr = optim_g.param_groups[0]["lr"]
                    losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
                    LOG.info(
                        "Train Epoch: {} [{:.0f}%]".format(
                            epoch, 100.0 * batch_idx / len(train_loader)
                        )
                    )
                    # LOG.info(
                    #    f"Losses: {[x.item() for x in losses]}, step: {global_step}, lr: {lr}"
                    # )

                    LOG.info(
                        "Discriminator Loss: {:f}, Generator Loss: {:f}, Feature Loss: {:f}, Mel Loss: {:f}, KL Loss: {:f}, Step: {:n}, LR: {:f}".format(
                            *([x.item() for x in losses] + [global_step, lr])
                        )
                    )

                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc_all,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                            "loss/g/kl": loss_kl,
                            "loss/g/lf0": loss_lf0,
                        }
                    )

                    # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                    # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                    # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                    image_dict = {
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(
                            y_mel[0].data.cpu().numpy()
                        ),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].data.cpu().numpy()
                        ),
                        "all/mel": utils.plot_spectrogram_to_numpy(
                            mel[0].data.cpu().numpy()
                        ),
                        "all/lf0": utils.plot_data_to_numpy(
                            lf0[0, 0, :].cpu().numpy(),
                            pred_lf0[0, 0, :].detach().cpu().numpy(),
                        ),
                        "all/norm_lf0": utils.plot_data_to_numpy(
                            lf0[0, 0, :].cpu().numpy(),
                            norm_lf0[0, 0, :].detach().cpu().numpy(),
                        ),
                    }

                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        images=image_dict,
                        scalars=scalar_dict,
                    )

                if global_step % hps.train.eval_interval == 0:
                    LOG.info("Saving checkpoints...")
                    if build_worst_loader and not use_worst_loader:
                        worst_loader.save()
                    evaluate(hps, net_g, eval_loader, writer_eval)
                    utils.save_checkpoint(
                        net_g,
                        optim_g,
                        hps.train.learning_rate,
                        epoch,
                        Path(hps.model_dir) / f"G_{global_step}.pth",
                    )
                    utils.save_checkpoint(
                        net_d,
                        optim_d,
                        hps.train.learning_rate,
                        epoch,
                        Path(hps.model_dir) / f"D_{global_step}.pth",
                    )
                    keep_ckpts = getattr(hps.train, "keep_ckpts", 0)
                    if keep_ckpts > 0:
                        utils.clean_checkpoints(
                            path_to_models=hps.model_dir,
                            n_ckpts_to_keep=keep_ckpts,
                            sort_by_time=True,
                        )

            global_step += 1
            # prof.step()

    if rank == 0:
        global start_time
        now = time.time()
        duration = format(now - start_time, ".2f")
        LOG.info(f"====> Epoch: {epoch}, time spend: {duration} s")
        start_time = now


def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    image_dict = {}
    audio_dict = {}
    with torch.no_grad():
        for batch_idx, items in enumerate(eval_loader):
            c, f0, spec, y, spk, _, uv = items
            g = spk[:1].cuda(0)
            spec, y = spec[:1].cuda(0), y[:1].cuda(0)
            c = c[:1].cuda(0)
            f0 = f0[:1].cuda(0)
            uv = uv[:1].cuda(0)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            try:
                y_hat = generator.module.infer(c, f0, uv, g=g)
            except:
                y_hat = generator.infer(c, f0, uv, g=g)

            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            audio_dict.update(
                {f"gen/audio_{batch_idx}": y_hat[0], f"gt/audio_{batch_idx}": y[0]}
            )
        image_dict.update(
            {
                "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
                "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy()),
            }
        )
    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
    )
    generator.train()


def fine_tune_embedding_freeze(net_d, net_g):
    net_d.requires_grad_(False)
    net_d.module.requires_grad_(False)
    for disc in net_d.module.multi_period_discriminator.discriminators:
        disc.conv_post.requires_grad_(True)
    for disc in net_d.module.multi_resolution_discriminator.discriminators:
        disc.conv_post.requires_grad_(True)
    for disc in net_d.module.multi_scale_discriminator.discriminators:
        disc.conv_post.requires_grad_(True)
    for pool in net_d.module.multi_scale_discriminator.meanpools:
        pool.requires_grad_(True)

    net_g.requires_grad_(False)
    net_g.module.requires_grad_(False)
    net_g.module.emb_g.requires_grad_(True)


# ENABLE_PROFILING = False
# PROFILE_MEMORY = False
# PROFILE_SHAPES = False
# SAVE_TRACE = True
# WARMUP_COUNT = 5
# _warmed_up = dict()
#
#
# def profile(record_function: str):
#     def decorator(func):
#         # Logic to disable profile wrapping entirely
#         # if not ENABLE_PROFILING:
#         #     return func
#         if not ENABLE_PROFILING:
#             def thin_wrapper(*args, **kwargs):
#                 with profiler.record_function(record_function):
#                     ret = func(*args, **kwargs)
#                 return ret
#
#             return thin_wrapper
#
#         def wrapper(*args, **kwargs):
#             with profiler.profile(record_shapes=PROFILE_SHAPES, profile_memory=PROFILE_MEMORY) as prof:
#                 with profiler.record_function(record_function):
#                     ret = func(*args, **kwargs)
#
#             # Logic to track warmup
#             if record_function not in _warmed_up.keys():
#                 _warmed_up[record_function] = 1
#                 return ret
#             _warmed_up[record_function] += 1
#             if _warmed_up[record_function] < WARMUP_COUNT:
#                 return ret
#
#             if SAVE_TRACE and _warmed_up[record_function] == WARMUP_COUNT:
#                 prof.export_chrome_trace(record_function + "_trace.json")
#             if PROFILE_MEMORY:
#                 print("CPU_MEM_USAGE")
#                 print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
#                 print("CUDA_MEM_USAGE")
#                 print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
#             print("CPU_TIME_TOTAL")
#             print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#             # print("CPU_TIME_TOTAL TOP LEVEL")
#             # print(prof.key_averages().table(sort_by="cpu_time_total", top_level_events_only=True, row_limit=10))
#             print("CUDA_TIME_USAGE")
#             print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#
#             return ret
#
#         return wrapper
#
#     return decorator
