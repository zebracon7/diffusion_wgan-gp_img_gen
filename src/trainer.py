from denoising_diffusion_pytorch import Trainer
import os

def create_trainer(diffusion, data_dir, results_folder):
    trainer = Trainer(
        diffusion,
        data_dir,
        train_batch_size=16,
        train_lr=2e-5,
        train_num_steps=50000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        results_folder=results_folder,
        save_and_sample_every=500,
        calculate_fid=False,
    )
    return trainer