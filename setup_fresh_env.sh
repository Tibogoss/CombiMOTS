#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-combimots}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_OPENCL_PATH="${CUDA_OPENCL_PATH:-}"
WITH_MGLTOOLS="${WITH_MGLTOOLS:-1}"
WITH_DOCKING_SETUP="${WITH_DOCKING_SETUP:-1}"
WITH_QUICKVINA_COMPILE="${WITH_QUICKVINA_COMPILE:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found on PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists."
  read -r -p "Remove and recreate it? [y/N] " answer
  case "$answer" in
    y|Y|yes|YES)
      conda env remove -n "$ENV_NAME" -y
      ;;
    *)
      echo "Aborting without modifying existing environment." >&2
      exit 1
      ;;
  esac
fi

conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
conda activate "$ENV_NAME"

if [[ "$WITH_MGLTOOLS" == "1" ]]; then
  conda install -c bioconda mgltools -y || {
    echo "Warning: mgltools could not be installed into '$ENV_NAME'. Install it separately if ligand PDBQT preparation is needed." >&2
  }
fi
conda install -c nvidia/label/cuda-11.7.0 cuda-nvcc -y
conda install -c nvidia cuda-opencl -y
conda install -c conda-forge ocl-icd-system -y
conda install -c conda-forge boost-cpp pdbfixer openbabel openmm rdkit 'zlib>=1.2.13' gxx_linux-64 -y

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -m pip install --upgrade pip
python -m pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
python -m pip install torch-geometric==2.0.4
python -m pip install -r requirements.txt
python -m pip install -e combimots/.

python -B -c "from setuptools import find_packages; print(find_packages('combimots'))"
python -B -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -B -c "import torch_geometric; print(torch_geometric.__version__)"
python -B -c "from rdkit import Chem; print(Chem.MolToSmiles(Chem.MolFromSmiles('CCO')))"
obabel -V
pmcts --help >/dev/null
combimots-preprocess --help >/dev/null
pmcts-validate-docking --help >/dev/null

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
  export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"

Run a pipeline dry-run with:
  combimots-preprocess --target-pair gsk3b_jnk3 --input-csv data/GSK3B_JNK3.csv --dry-run

Optional flags:
  WITH_MGLTOOLS=0 bash setup_fresh_env.sh $ENV_NAME
  WITH_DOCKING_SETUP=0 bash setup_fresh_env.sh $ENV_NAME
  WITH_QUICKVINA_COMPILE=1 bash setup_fresh_env.sh $ENV_NAME

EOF
