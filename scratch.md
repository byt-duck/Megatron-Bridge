# setup
1. 
docker build \
  -f Dockerfile.cuda-compat \
  -t nemo:26.08.01-cuda-compat-r550 \
  .

2. 
git submodule update --init

docker volume create mbridge-uv-cache
docker volume create mbridge-hf-cache
docker volume create mbridge-outputs

docker create \
  --name mbridge-dev \
  --gpus all \
  --shm-size=24g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -w /workdir \
  -v "$PWD:/workdir" \
  -v mbridge-uv-cache:/opt/uv_cache \
  -v mbridge-hf-cache:/root/.cache/huggingface \
  -v mbridge-outputs:/outputs \
  nemo:26.08.01-cuda-compat-r550 \
  bash

3. 
docker start mbridge-dev
docker exec -it mbridge-dev bash

4. 
cd /workdir
git submodule update --init

TMS_CUDA_MAJOR=13 uv sync \
  --link-mode copy \
  --locked \
  --all-extras \
  --all-groups \
  --no-group diffusion \
  --no-install-package mamba-ssm \
  --no-install-package nvidia-cudnn-frontend \
  --no-install-package transformer-engine \
  --no-install-package transformer-engine-torch

uv pip install --no-deps --reinstall -e 3rdparty/Megatron-LM

5. [nano-pretrain smoke]
uv run --no-sync python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  examples/models/nemotron/nemotron_3/nano/pretrain_nemotron_3_nano.py \
  model.seq_length=256 \
  dataset.seq_length=256 \
  model.cuda_graph_impl=none \
  optimizer.optimizer_cpu_offload=true \
  optimizer.optimizer_offload_fraction=1.0 \
  optimizer.overlap_cpu_optimizer_d2h_h2d=true \
  train.train_iters=1 \
  train.global_batch_size=8 \
  train.micro_batch_size=1 \
  scheduler.lr_warmup_iters=0 \
  scheduler.lr_decay_iters=1 \
  validation.eval_iters=0 \
  validation.eval_interval=0 \
  validation.eval_global_batch_size=8 \
  validation.eval_micro_batch_size=1 \
  checkpoint.load=null \
  checkpoint.save=null \
  logger.tensorboard_dir=null \
  logger.log_interval=1


