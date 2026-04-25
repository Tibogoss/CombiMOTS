#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-combimots}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
RECREATE_ENV="${RECREATE_ENV:-0}"
WITH_MGLTOOLS="${WITH_MGLTOOLS:-0}"
WITH_DOCKING_SETUP="${WITH_DOCKING_SETUP:-1}"
WITH_QUICKVINA_COMPILE="${WITH_QUICKVINA_COMPILE:-0}"
CUDA_OPENCL_PATH="${CUDA_OPENCL_PATH:-}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [[ "$RECREATE_ENV" == "1" ]]; then
    conda env remove -n "$ENV_NAME" -y
  else
    echo "Conda environment '$ENV_NAME' already exists." >&2
    echo "Set RECREATE_ENV=1 to remove and recreate it." >&2
    exit 1
  fi
fi

conda create -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION" pip uv 'setuptools<81' wheel -y
conda activate "$ENV_NAME"

# Keep binary/system packages in conda. These are intentionally not installed by uv.
conda install -c nvidia/label/cuda-11.7.0 cuda-nvcc -y
conda install -c nvidia cuda-opencl -y
conda install -c conda-forge ocl-icd-system clinfo -y
conda install -c conda-forge boost-cpp pdbfixer openbabel openmm rdkit 'zlib>=1.2.13' gxx_linux-64 -y

if [[ "$WITH_MGLTOOLS" == "1" ]]; then
  conda install -c bioconda mgltools -y || {
    echo "Warning: mgltools could not be installed into '$ENV_NAME'. Install it separately if receptor preparation is needed." >&2
  }
fi

if [[ -z "${OCL_ICD_VENDORS:-}" ]]; then
  if [[ -d /etc/OpenCL/vendors ]]; then
    export OCL_ICD_VENDORS=/etc/OpenCL/vendors
  elif [[ -d "$CONDA_PREFIX/etc/OpenCL/vendors" ]]; then
    export OCL_ICD_VENDORS="$CONDA_PREFIX/etc/OpenCL/vendors"
  fi
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CONDA_PREFIX/lib"
UV_PYTHON="$CONDA_PREFIX/bin/python"

# Install CUDA/PyG wheels explicitly before the pure-Python layer.
uv pip install --python "$UV_PYTHON" torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
uv pip install --python "$UV_PYTHON" torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
uv pip install --python "$UV_PYTHON" torch-geometric==2.0.4

# Do not use `uv pip sync` yet: it can remove CUDA/PyG wheels unless the lock strategy includes alternate indexes.
uv pip install --python "$UV_PYTHON" -r requirements.txt
uv pip install --python "$UV_PYTHON" --no-deps -e .

python -B -c "from setuptools import find_packages; print(find_packages('combimots'))"
python -B -c "import pmcts, preprocess; print(pmcts.__version__)"
python -B -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -B -c "import torch_geometric; print(torch_geometric.__version__)"
python -B -c "from rdkit import Chem; print(Chem.MolToSmiles(Chem.MolFromSmiles('CCO')))"
python -B -c "import chemprop; print('chemprop import ok')"
obabel -V
opencl_platforms="$(clinfo -l 2>/dev/null || true)"
if [[ -n "$opencl_platforms" ]]; then
  printf '%s\n' "$opencl_platforms"
else
  echo "Warning: no OpenCL platform detected. Docking needs clinfo to show an NVIDIA CUDA platform." >&2
fi
pmcts --help >/dev/null
chemprop_train --help >/dev/null
chemprop_predict --help >/dev/null
combimots-preprocess --help >/dev/null
pmcts-validate-docking --help >/dev/null
uv pip check --python "$UV_PYTHON"

if [[ "$WITH_DOCKING_SETUP" == "1" ]]; then
  python setup_docking.py

  if [[ "$WITH_QUICKVINA_COMPILE" == "1" ]]; then
    quickvina_dir="combimots/pmcts/docking/Vina-GPU-2.1/QuickVina2-GPU-2.1"
    if [[ -n "$CUDA_OPENCL_PATH" ]]; then
      make -C "$quickvina_dir" source OPENCL_LIB_PATH="$CUDA_OPENCL_PATH"
    else
      make -C "$quickvina_dir" source
    fi
  fi

  pmcts-validate-docking --target-pair gsk3b_jnk3 || true
fi

cat <<EOF

Environment '$ENV_NAME' is ready.

Activate it with:
  conda activate $ENV_NAME
  if [ -d /etc/OpenCL/vendors ]; then export OCL_ICD_VENDORS=/etc/OpenCL/vendors; fi
  export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:+\$LD_LIBRARY_PATH:}\$CONDA_PREFIX/lib"

Run a pipeline dry-run with:
  combimots-preprocess --target-pair gsk3b_jnk3 --input-csv data/GSK3B_JNK3.csv --dry-run

Optional flags:
  RECREATE_ENV=1 bash setup_fresh_env.sh $ENV_NAME
  WITH_MGLTOOLS=1 bash setup_fresh_env.sh $ENV_NAME
  WITH_DOCKING_SETUP=0 bash setup_fresh_env.sh $ENV_NAME
  WITH_QUICKVINA_COMPILE=1 bash setup_fresh_env.sh $ENV_NAME

EOF
