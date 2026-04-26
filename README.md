<h1 align="center">CombiMOTS: Combinatorial Multi-Objective Tree Search for Dual-Target Molecule Generation</h1>
<p align="center">
    <a href="https://openreview.net/forum?id=FSlTEObdLl"><img src="https://img.shields.io/badge/OpenReview-ICML'25.16227-b31b1b.svg" alt="Paper"></a>
    <a href="https://icml.cc/media/PosterPDFs/ICML%202025/45885.png?t=1752232241.6172879"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a>
    <a href="./assets/ICML2025-CombiMOTS_Slides.pdf"> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a>
</p>
Official implementation of CombiMOTS for Fragment-based Monte Carlo Tree Search for Dual-Inhibitors Molecular Graph Generation.

Refer to `Poster` or `Slides` for a more in-depth overview of our work!

<p align="center"><img src="./assets/overview.png" width=80%></p>
<p align="center">Project overview.</p>

### Broader Applications

- We release a pretrained ensemble ChemProp ClinTox model in `models/clintox` that can be used for Toxicity Optimization as described in our original manuscript.
- We created another repository (<a href="https://github.com/Tibogoss/KinSel"> <img src="https://img.shields.io/badge/KinSel-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="KinSel"> </a>) using CombiMOTS for the Selective Molecular Generation (using CDK7 as the target). Our main manuscript also discusses motivation background and implementation details.

# Baseline papers
Activity-aware fragments are obtained with Graph Information Bottleneck - Adapted from https://arxiv.org/abs/2310.00841

Our Pareto MCTS pipeline is adapted from **SyntheMol** https://www.nature.com/articles/s42256-024-00809-7

The 13 **Enamine** (https://enamine.net/) REAL Space and corresponding reactions are also provided by the work above.

To accelerate molecular docking simulation, we utilize **QuickVina-GPU-2.1** from https://pubmed.ncbi.nlm.nih.gov/39320991/


# Install Environment
Implementation was originally conducted with Python 3.10 and CUDA 11.7. The setup script creates the conda environment, installs Python dependencies with uv, and installs CombiMOTS editable from `combimots/.`.

```sh
bash setup_fresh_env.sh combimots
```

On HPC systems, load the CUDA/NVIDIA module before activating the environment:

```sh
conda deactivate
unset LD_LIBRARY_PATH OCL_ICD_VENDORS
module load CUDA/12.8.0  # or the CUDA module provided by your cluster
```

Activate CombiMOTS and expose the NVIDIA OpenCL ICD:

```sh
conda activate combimots
if [ -d /etc/OpenCL/vendors ]; then export OCL_ICD_VENDORS=/etc/OpenCL/vendors; fi
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CONDA_PREFIX/lib
```

QED/SA objectives require PyTDC, which is optional because the default pipeline does not need it and PyTDC can pull Python cheminformatics dependencies that conflict with the conda RDKit stack. Install it only when running `pmcts --qed_sa` or `pmcts --all_objectives`:

```sh
uv pip install --python "$CONDA_PREFIX/bin/python" pytdc
python -B -c "from tdc import Oracle; print(Oracle(name='qed')('CCO'))"
```

## Docking Setup

Clone/configure QuickVina-GPU, compile it, and validate the docking setup:

```sh
python setup_docking.py
(cd combimots/pmcts/docking/Vina-GPU-2.1/QuickVina2-GPU-2.1 && make source)
pmcts-validate-docking --target-pair gsk3b_jnk3
```

Check OpenCL before running docking-heavy steps:

```sh
clinfo -l
```

# Note to the user
The next sections describe all pre-processing steps (running scripts 0 to 8).

If you only want to run generation and evaluation, **we provide processed data and model checkpoints**.
You may skip these steps and directly go to the generation section.

The preprocessing runner can run the full sequence while keeping the numbered scripts available:

```sh
combimots-preprocess \
  --target-pair gsk3b_jnk3 \
  --input-csv data/GSK3B_JNK3.csv
```

Run a step range with:

```sh
combimots-preprocess --target-pair gsk3b_jnk3 --input-csv data/GSK3B_JNK3.csv --from-step similar-blocks --to-step filter-reactions --resume
```

# Pipeline

In `/data` you may place a .csv file containing:
- smiles
- {target1}_activity
- {target2}_activity

For demonstration, we provide data for the GSK3B-JNK3, EGFR-MET and PIK3CA-mTOR target pairs.

This data is curated from **ExCAPE-DB v2** (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0203-5)

# Train Chemprop (D-MPNN) Property Predictor
```sh
chemprop_train --data_path {data_path} \
--dataset_type classification \
--split_type cv \
--num_folds 10 \
--seed 42 \
--gpu 0 \
--save_dir models/${model_name}

# $ chemprop_train --data_path data/GSK3B_JNK3.csv --dataset_type classification --split_type cv --num_folds 10 --seed 42 --gpu 0 --save_dir models/gsk3b_jnk3
```

# Data Preparation

## Search Space Reduction

### Fragment-based Graph Information Bottleneck 

Process the .csv to a .pt:
```sh
python utils_fgib/data.py --csv_path $data/{YOUR_DATA.csv} --target ${activity}

# $ python utils_fgib/data.py --csv_path data/GSK3B_JNK3.csv --target gsk3b_activity
# $ python utils_fgib/data.py --csv_path data/GSK3B_JNK3.csv --target jnk3_activity

```
Train the modules:
```sh
python 1-train_fgib.py -g ${gpu_id} --target ${target_activity}

# $ python 1-train_fgib.py -g 0 --target gsk3b_activity
# $ python 1-train_fgib.py -g 0 --target jnk3_activity
```

Extract and save the building blocks for both targets:
```sh
python 2-get_frags.py -g ${gpu_id} -t ${target_activity} -m ${target_pt_path} -v ${frags_path}

# $ python 2-get_frags.py -g 0 -t gsk3b_activity -m ckpt/gsk3b_activity_10.pt -v data/gsk3b.txt
# $ python 2-get_frags.py -g 0 -t jnk3_activity -m ckpt/jnk3_activity_10.pt -v data/jnk3.txt
```

Clean and merge both targets' building blocks:
```sh
python 3-frags_to_blocks.py ${frags_path1} ${frags_path2} ${fragments_path}

# $ python 3-frags_to_blocks.py data/gsk3b.txt data/jnk3.txt data/fgib_frags.csv
```

### Map FGIB fragments to Enamine's REAL Space Building Blocks
```sh
python 4-get_similar_blocks.py --custom_path ${fragments_path} \
--real_path combimots/pmcts/resources/real/building_blocks.csv \
--output_path models/${model_name}/${blocks_path} \
--threshold ${tanim_thresh} --batch_size ${bs}

# $ python 4-get_similar_blocks.py --custom_path data/fgib_frags.csv --real_path combimots/pmcts/resources/real/building_blocks.csv --output_path models/gsk3b_jnk3/similar.csv --threshold 0.4 --batch_size 2500
```
Remove Salts:
```sh
chemfunc canonicalize_smiles --data_path models/${model_name}/${blocks_path} \
--save_path models/${model_name}/${blocks_path} \
--remove_salts --delete_disconnected_mols

# $ chemfunc canonicalize_smiles --data_path models/gsk3b_jnk3/similar.csv --save_path models/gsk3b_jnk3/similar.csv --remove_salts --delete_disconnected_mols
```
Remove Br, Si and Li atoms for **QuickVina-GPU-2.1** compatibility:
```sh
python 5-remove_B_Si_Li_blocks.py models/${model_name}/${blocks_path} models/${model_name}/${blocks_path}

# $ python 5-remove_B_Si_Li_blocks.py models/gsk3b_jnk3/similar.csv models/gsk3b_jnk3/similar.csv
```
### Precompute activities and docking scores

Bioactivities using **Chemprop** (D-MPNN):
```sh
chemprop_predict --test_path models/${model_name}/${blocks_path} \
--preds_path models/${model_name}/${blocks_path} \
--checkpoint_dir models/${model_name}

# $ chemprop_predict --test_path models/gsk3b_jnk3/similar.csv --preds_path models/gsk3b_jnk3/precompute.csv --checkpoint_dir models/gsk3b_jnk3
```
Docking Scores using **QuickVina-GPU-2.1**
This step is very important as docking oracles are the most expensive components during generation.
```sh
python 6-precompute_docking_scores.py models/${model_name}/${blocks_path} models/${model_name}/${blocks_path} --target_pair ${target_pair}

# $ python 6-precompute_docking_scores.py models/gsk3b_jnk3/precompute.csv models/gsk3b_jnk3/final_blocks.csv --target_pair gsk3b_jnk3
# target_pair: str= ["gsk3b_jnk3", "egfr_met", "pik3ca_mtor", "dhodh_rorgt"]
```
### Map building blocks to Enamine's reactions

```sh
# Map building blocks -> .pkl
python 7-map_bbs_to_search_space.py --input models/${model_name}/${blocks_path} \
--real_path combimots/pmcts/resources/real/reaction_to_building_blocks.pkl \
--save_path combimots/pmcts/resources/real/${target_pair}.pkl \
--smiles_column smiles

# $ python 7-map_bbs_to_search_space.py --input models/gsk3b_jnk3/final_blocks.csv --real_path combimots/pmcts/resources/real/reaction_to_building_blocks.pkl --save_path combimots/pmcts/resources/real/gsk3b_jnk3.pkl --smiles_column smiles

# Filter non-matching BBs w.r.t. the provided reacions
python 8-filter_reactions_to_blocks.py \
--reaction_to_building_blocks_path combimots/pmcts/resources/real/${target_pair}.pkl \
--save_path combimots/pmcts/resources/real/${target_pair}.pkl

# $ python 8-filter_reactions_to_blocks.py --reaction_to_building_blocks_path combimots/pmcts/resources/real/gsk3b_jnk3.pkl --save_path combimots/pmcts/resources/real/gsk3b_jnk3.pkl
```

# Generation: Pareto Monte-Carlo Tree Search

Generated CSV files store `dockingscore1` and `dockingscore2` as the transformed maximization objectives used by MCTS (`-raw_docking_score / 20`). Raw QuickVina energies are also saved as `dockingscore1_raw` and `dockingscore2_raw`.

```sh
pmcts --model_path models/${model_name} --save_dir generations/${model_name}/ \
--target_activities ${activity1, activity2} \
--target_pair ${target_pair} \
--building_blocks_path models/${model_name}/final_blocks.csv \
--n_rollout 5000

# $ pmcts --model_path models/${model_name} --save_dir generations/${model_name}/ \
--target_activities gsk3b_activity jnk3_activity \
--target_pair gsk3b_jnk3 \
--building_blocks_path models/gsk3b_jnk3/final_blocks.csv \
--n_rollout 5000
```

## Evaluation

Filter out the molecules predicted as dual actives
```sh
python 9-filter_dual_actives.py generations/${model_name}/pareto_molecules.csv generations/${model_name}/pareto_dual_actives.csv

# $ python 9-filter_dual_actives.py generations/gsk3b_jnk3/pareto_molecules.csv generations/gsk3b_jnk3/pareto_dual_actives.csv
```
Optionally, re-docking simulations have to be run separately.

For all other metrics (Validity, Uniqueness, Novelty, Diversity, Avg.QED, Avg.SA), we provide `evaluate.py`:
```sh
python 10-evaluate.py --model models/${model_name} \
--generation generations/${model_name}/pareto_dual_actives.csv \
--training {path_to_dual_positives_of_training_set_csv}

# $ python 10-evaluate.py --generation generations/gsk3b_jnk3/pareto_dual_actives.csv --training data/GSK3B_dual_actives.csv
```

If you find our paper/repo useful or use it for personal projects/research, please cite our original paper: -->

```bibtex -->
@inproceedings{
southiratn2025combimots,
title={Combi{MOTS}: Combinatorial Multi-Objective Tree Search for Dual-Target Molecule Generation},
author={Thibaud Southiratn and Bonil Koo and Yijingxiu Lu and Sun Kim},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=FSlTEObdLl}
}
```
