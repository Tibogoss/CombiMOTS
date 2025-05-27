""" Adapted from the implementation of:
https://github.com/swansonk14/SyntheMol/ """

"""Contains training and predictions functions for Chemprop models."""
from pathlib import Path
from typing import List

import numpy as np
import torch
from chemprop.models import MoleculeModel
from chemprop.utils import load_checkpoint, load_scalers
from sklearn.preprocessing import StandardScaler


def chemprop_load(
        model_path: Path,
        device: torch.device = torch.device('cpu')
) -> MoleculeModel:
    """Loads a Chemprop model.

    :param model_path: A path to a Chemprop model.
    :param device: The device on which to load the model.
    :return: A Chemprop model.
    """
    return load_checkpoint(
        path=str(model_path),
        device=device
    ).eval()


def chemprop_load_scaler(
        model_path: Path
) -> StandardScaler:
    """Loads a Chemprop model's data scaler.

    :param model_path: A path to a Chemprop model.
    :return: A data scaler.
    """
    return load_scalers(path=str(model_path))[0]


def chemprop_predict_on_molecule(
        model: MoleculeModel,
        smiles: str,
        scaler: StandardScaler | None = None
) -> List[float]:
    """Predicts properties for a molecule using a Chemprop model.

    :param model: A Chemprop model.
    :param smiles: A SMILES string.
    :param scaler: A data scaler (if applicable).
    :return: Prediction on the molecule.
    """
    out = model(
        batch=[[smiles]],
        features_batch=None
    ).tolist()

    if scaler is not None:
        out[0][0] = scaler.inverse_transform([out[0][0]])[0][0]
        out[0][1] = scaler.inverse_transform([out[0][1]])[0][0]

    return [float(x) for x in out[0]]


def chemprop_predict_on_molecule_ensemble(
        models: list[MoleculeModel],
        smiles: str,
        scalers: list[StandardScaler] | None = None,
) -> List[float]:
    """Predicts properties for a molecule using an ensemble of Chemprop models.

    :param models: An ensemble of Chemprop models.
    :param smiles : A SMILES string.
    :param scalers: An ensemble of data scalers (if applicable).
    :return: Ensemble prediction for a molecule as [mean(activity1), mean(activity2)].
    """
    # Get predictions from each model
    predictions = [
        chemprop_predict_on_molecule(
            model=model,
            smiles=smiles,
            scaler=scaler
        ) for model, scaler in zip(models, scalers or [None] * len(models))
    ]
    
    # Convert to numpy array for easier manipulation
    predictions = np.array(predictions)
    
    return [
        float(np.mean(predictions[:, 0])),
        float(np.mean(predictions[:, 1]))
    ]
