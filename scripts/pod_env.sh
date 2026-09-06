#!/usr/bin/env bash
# Pod environment. Source this before installing or downloading anything:
#
#   source /workspace/repo/scripts/pod_env.sh
#
# And make it stick across sessions:
#
#   echo 'source /workspace/repo/scripts/pod_env.sh' >> ~/.bashrc
#
# Why this matters: HuggingFace and uv default their caches to ~/.cache, which
# lives on the 40 GB container disk. The model weights alone are ~54 GB, so a
# download would fail on space -- and anything that did fit would be erased when
# the pod stops. Both caches belong on the network volume.

export HF_HOME=/workspace/hf_cache
export UV_CACHE_DIR=/workspace/uv_cache

# Where generated traces land. Gitignored; pushed to a private HF dataset repo
# after each completed run so the volume is not a single point of failure.
export TIR_OUTPUTS=/workspace/outputs

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TIR_OUTPUTS"

# The gen venv lives on the container disk, not /workspace -- it is ~30k small
# files and /workspace is a network filesystem (MooseFS). Wiped when the pod
# stops, but rebuilds in a couple of minutes from the cache above.
export VENV_GEN=/root/venv-gen

echo "HF_HOME       = $HF_HOME"
echo "UV_CACHE_DIR  = $UV_CACHE_DIR"
echo "TIR_OUTPUTS   = $TIR_OUTPUTS"
echo "VENV_GEN      = $VENV_GEN"
