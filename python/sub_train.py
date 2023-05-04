'''
usage: train.py [-h] [--hidden HIDDEN] [--latent-dim LATENT_DIM]
                [--no-full-cov] [--zero-inflate] [--epochs EPOCHS]
                [--seed SEED] [--batch-size BATCH_SIZE] [--lr LR]
                [--subsample-rows SUBSAMPLE_ROWS] [--add-log1p-l2-recon-loss]

options:
  -h, --help            show this help message and exit
  --hidden HIDDEN
  --latent-dim LATENT_DIM
  --no-full-cov
  --zero-inflate
  --epochs EPOCHS
  --seed SEED
  --batch-size BATCH_SIZE
  --lr LR
  --subsample-rows SUBSAMPLE_ROWS
  --add-log1p-l2-recon-loss
'''
import subprocess
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

EPOCHS = [1]
latents = [16, 32, 128, 256]
'''
    [256],
    [256, 64],
'''
hiddens = [
    [512, 256],
    # [-1, 512, 256],
    # [-1, 256],
    # [1024, 512, 256],
    # [512, 512, 256],
    [256, 256, 128],
]

full_covs = [False, True]

zero_inflate = [False]

for epoch in EPOCHS:
    for latent in latents:
        for hidden in hiddens:
            hidden_set = hidden[:]
            hidden_set[-1] = max(latent, hidden_set[-1])
            for full_cov in full_covs:
                hidstr = ",".join(map(str, hidden_set))
                covstr = "" if full_cov else "--no-full-cov"
                cov = "full" if full_cov else "diag"
                outpath = f"{hidstr}.{latent}.{cov}.log"
                if len(hidden_set) <= 1:
                    batch_size = 8
                else:
                    batch_size = 2
                cmd = f"python3 train.py --epochs {epoch} --hidden {hidstr} --latent-dim {latent} --batch-size {batch_size} {covstr} -G16 &> {outpath}"
                try:
                    subprocess.check_call(cmd, shell=True)
                except KeyboardInterrupt:
                    break
                except:
                    print(cmd, "Failed. Continuing")
                    continue
