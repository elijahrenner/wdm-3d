# Optuna hyperparameter search configuration
GPU=0
SEED=42
CHANNELS=64
DATASET='inpaint'
MODEL='ours_unet_128'
MODALITIES=1
N_TRIALS=20
VAL_INTERVAL=1000
EARLY_STOPPING_PATIENCE=5
DROPOUT=0.1

# architecture settings
if [[ $MODEL == 'ours_unet_128' ]]; then
  CHANNEL_MULT=1,2,2,4,4
  IMAGE_SIZE=128
  ADDITIVE_SKIP=True
  USE_FREQ=False
  BATCH_SIZE=10
elif [[ $MODEL == 'ours_unet_256' ]]; then
  CHANNEL_MULT=1,2,2,4,4,4
  IMAGE_SIZE=256
  ADDITIVE_SKIP=True
  USE_FREQ=False
  BATCH_SIZE=1
elif [[ $MODEL == 'ours_wnet_128' ]]; then
  CHANNEL_MULT=1,2,2,4,4
  IMAGE_SIZE=128
  ADDITIVE_SKIP=False
  USE_FREQ=True
  BATCH_SIZE=10
elif [[ $MODEL == 'ours_wnet_256' ]]; then
  CHANNEL_MULT=1,2,2,4,4,4
  IMAGE_SIZE=256
  ADDITIVE_SKIP=False
  USE_FREQ=True
  BATCH_SIZE=1
else
  echo "MODEL TYPE NOT FOUND -> Check the supported configurations again"
  exit 1
fi

# dataset settings
DATASET_IMAGE_SIZE=256
DESIRED_IMAGE_SIZE=128
if [[ $DATASET == 'brats' ]]; then
  DATA_DIR=~/wdm-3d/data/BRATS/
elif [[ $DATASET == 'lidc-idri' ]]; then
  DATA_DIR=~/wdm-3d/data/LIDC-IDRI/
  IN_CHANNELS=8
elif [[ $DATASET == 'inpaint' ]]; then
  DATA_DIR=/workspace/sts/data
  IN_CHANNELS=$(( MODALITIES * 8 + 8 ))
  OUT_CHANNELS=8
else
  echo "DATASET NOT FOUND -> Check the supported datasets again"
  exit 1
fi

COMMON="
--dataset=${DATASET}
--num_channels=${CHANNELS}
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=${CHANNEL_MULT}
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=${BATCH_SIZE}
--num_groups=32
--in_channels=${IN_CHANNELS}
--out_channels=${OUT_CHANNELS:-$IN_CHANNELS}
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=${ADDITIVE_SKIP}
--use_freq=${USE_FREQ}
--dropout=${DROPOUT}
--val_interval=${VAL_INTERVAL}
--predict_xstart=True
--early_stopping
--early_stopping_patience=${EARLY_STOPPING_PATIENCE}
"

TRAIN="
--data_dir=${DATA_DIR}
--resume_checkpoint=
--resume_step=0
--image_size=${IMAGE_SIZE}
--dataset_image_size=${DATASET_IMAGE_SIZE}
--desired_image_size=${DESIRED_IMAGE_SIZE}
--use_fp16=False
--lr=1e-5
--save_interval=100000
--num_workers=12
--devices=${GPU}
--seed=${SEED}
"

python scripts/hyperparameter_search.py $TRAIN $COMMON --n_trials=${N_TRIALS}
