#!/usr/bin/env bash
# Pod environment. Source this before installing or downloading anything:
#
#   source /workspace/repo/scripts/pod_env.sh
#
# And make it stick across sessions:
#
#   echo 'source /workspace/repo/scripts/pod_env.sh' >> ~/.bashrc
#
# Every export here dies with the shell, which is why they live in a file. A
# browser refresh gives you a new shell with none of them set -- that is what
# causes "Permission denied (publickey)" on git push out of nowhere.

# --- caches -----------------------------------------------------------------
# HuggingFace and uv default to ~/.cache, which is on the 40 GB container disk
# and is erased when the pod stops. Model weights alone are ~54 GB.
export HF_HOME=/workspace/hf_cache
export UV_CACHE_DIR=/workspace/uv_cache

# Cache and venv are on different filesystems (MooseFS vs overlay), so uv cannot
# hardlink between them and falls back to copying. Expected; this silences it.
export UV_LINK_MODE=copy

# Where generated traces land. Gitignored; pushed to a private HF dataset repo
# after each completed run so the volume is not a single point of failure.
export TIR_OUTPUTS=/workspace/outputs

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TIR_OUTPUTS"

# --- gen venv ---------------------------------------------------------------
# On the container disk, not /workspace: it is ~30k small files and /workspace
# is network storage. Wiped when the pod stops, but rebuilds in ~2 min:
#   uv venv $VENV_GEN --python 3.11
#   uv pip install --python $VENV_GEN -r envs/gen.lock.txt
export VENV_GEN=/root/venv-gen

# --- analysis venv ----------------------------------------------------------
# TransformerLens + EAP-IG, for every circuits/ stage. Same reasoning as VENV_GEN:
# container disk, wiped on stop, rebuilt with
#   uv venv $VENV_ANALYSIS --python 3.11
#   uv pip install --python $VENV_ANALYSIS -r envs/analysis.txt -e .
# Kept separate from VENV_GEN because the two pin conflicting torch versions.
export VENV_ANALYSIS=/root/venv-analysis

# --- github -----------------------------------------------------------------
# The durable copy of the key is on /workspace, but SSH refuses to use it there:
# MooseFS ignores chmod, so the key stays world-readable and SSH skips it. Copy
# to the container disk, where permissions actually apply.
if [ -f /workspace/.ssh/github ] && [ ! -f ~/.ssh/github ]; then
    mkdir -p ~/.ssh
    cp /workspace/.ssh/github ~/.ssh/github
    chmod 600 ~/.ssh/github
fi

# SSH only looks at default key names (id_rsa, id_ed25519). Ours is "github",
# so without -i it is never found.
export GIT_SSH_COMMAND="ssh -i ~/.ssh/github -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

# Commit authorship -- a different thing from the authentication above, and
# stored in ~/.gitconfig, which lives on the container disk and so vanishes with
# every new pod. Without it the first commit fails with "Author identity unknown".
git config --global user.name "Rohith Reddy Mandala" 2>/dev/null
git config --global user.email "rohithreddy24242@gmail.com" 2>/dev/null

# --- huggingface token ------------------------------------------------------
# Optional. Lives on the volume, never in git. Raises download rate limits and
# is required to push traces to the private dataset repo.
#   echo 'export HF_TOKEN=hf_xxx' > /workspace/.hf_token && chmod 600 /workspace/.hf_token
[ -f /workspace/.hf_token ] && source /workspace/.hf_token

echo "HF_HOME      = $HF_HOME"
echo "UV_CACHE_DIR = $UV_CACHE_DIR"
echo "TIR_OUTPUTS  = $TIR_OUTPUTS"
echo "VENV_GEN     = $VENV_GEN $([ -x $VENV_GEN/bin/python ] && echo '(built)' || echo '(not built)')"
echo "VENV_ANALYSIS= $VENV_ANALYSIS $([ -x $VENV_ANALYSIS/bin/python ] && echo '(built)' || echo '(not built)')"
# Never interpolate $HF_TOKEN itself here -- ${HF_TOKEN:-not set} expands to the
# token value when it is set, which prints the secret to the terminal.
echo "HF_TOKEN     = $([ -n "$HF_TOKEN" ] && echo ok || echo missing)"
echo "git ssh key  = $([ -f ~/.ssh/github ] && echo ok || echo MISSING)"
