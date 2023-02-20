import os
import json
import numpy as np
import pandas as pd
import argparse
import parse
import torch
import joblib
from tqdm import tqdm
from torch import nn
from torchvision import transforms

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error

from multimodaldatasets.datasets import DataManager
from augmentations import Normalize, Reshape, Transformer
from surfify.utils import icosahedron, downsample, downsample_data
from surfify.models import SphericalHemiFusionEncoder



parser = argparse.ArgumentParser(description="Spherical predictor")
parser.add_argument(
    "--data", default="hcp", choices=("hbn", "euaims", "hcp", "openbhb", "privatebhb"),
    help="the input cohort name.")
parser.add_argument(
    "--datadir", metavar="DIR", help="data directory path.", required=True)
parser.add_argument(
    "--outdir", metavar="DIR", help="output directory path.", required=True)
parser.add_argument(
    "--to-predict", default="age",
    help="the name of the variable to predict.")
parser.add_argument(
    "--method", default="regression", choices=("regression", "classification"),
    help="the prediction method.")
parser.add_argument(
    "--setups-file", default="None",
    help="the path to the file linking the setups to the pretrained encoder's path.")
args = parser.parse_args()


modalities = ["surface-lh", "surface-rh"]
metrics = ["thickness", "curv", "sulc"]
n_features = len(metrics)
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encoder_cp_from_model_cp(checkpoint):
    name_to_check = "backbone"
    checkpoint = {".".join(key.split(".")[1:]): value 
                for key, value in checkpoint["model_state_dict"].items() if name_to_check in key}
    return checkpoint

def params_from_args(params):
    old_format = (
        "deepint_barlow_{}_surf_{}_features_fusion_{}_act_{}_bn_{}_conv_{}"
        "_latent_{}_wd_{}_{}_epochs_lr_{}_bs_{}_ba_{}_ima_{}_gba_{}_cutout_{}"
        "_normalize_{}_standardize_{}")
    new_format = (
        "pretrain_{}_on_{}_surf_order_{}_with_{}_features_fusion_{}_act_{}"
        "_bn_{}_conv_{}_latent_{}_wd_{}_{}_epochs_lr_{}_reduced_{}_bs_{}_ba_{}"
        "_ima_{}_blur_{}_noise_{}_cutout_{}_normalize_{}_standardize_{}_"
        "loss_param_{}_sigma_{}")
    old = True
    try:
        parsed = parse.parse(old_format, params.split("/")[-2])
    except Exception:
        old = False
        parsed = parse.parse(new_format, params)
    args_names = [
        "data_train", "n_features", "fusion_level", "activation",
        "batch_norm", "conv_filters", "latent_dim",
        "weight_decay", "epochs", "learning_rate", "batch_size",
        "batch_augment", "inter_modal_augment", "gaussian_blur_augment",
        "cutout", "normalize", "standardize"]
    if not old:
        args_names = [
            "algo", "data_train", "ico_order", "n_features", "fusion_level",
            "activation", "batch_norm", "conv_filters",
            "latent_dim", "weight_decay", "epochs", "learning_rate",
            "reduce_lr", "batch_size", "batch_augment",
            "inter_modal_augment", "blur", "noise", "cutout",
            "normalize", "standardize", "loss_param", "sigma"]
    for idx, value in enumerate(parsed.fixed):
        if value.isdigit():
            value = float(value)
            value = int(value) if int(value) == value else value
        elif value in ["True", "False"]:
            value = value == "True"
        setattr(args, args_names[idx], value)
    return args


setups = pd.read_table(args.setups_file)

ico_order = 5

input_shape = (len(metrics), len(icosahedron(ico_order)[0]))

order = 7
ico_verts, _ = icosahedron(order)
down_indices = []
for low_order in range(order - 1, ico_order - 1, -1):
    low_ico_verts, _ = icosahedron(low_order)
    down_indices.append(downsample(ico_verts, low_ico_verts))
    ico_verts = low_ico_verts
def transform(x):
    downsampled_data = downsample_data(x, 7 - ico_order, down_indices)
    return np.swapaxes(downsampled_data, 1, 2)


kwargs = {
    "surface-rh": {"metrics": metrics},
    "surface-lh": {"metrics": metrics},
}

if args.data in ["hbn", "euaims"]:
    kwargs["surface-lh"]["symetrized"] = True
    kwargs["surface-rh"]["symetrized"] = True

    modalities.append("clinical")

if "clinical" in modalities:
    clinical_names = np.load(
        os.path.join(args.datadir, "clinical_names.npy"),allow_pickle=True)
    if args.to_predict in clinical_names:
        index_to_predict = clinical_names.tolist().index(args.to_predict)

evaluation_metrics = {"mae": mean_absolute_error, "r2": r2_score}
final_metric = "mae"
what_is_best = {"mae": "lower", "r2": "higher"}


for setup_id in setups["id"].values:
    params = setups[setups["id"] == setup_id]["args"].values[0]
    cp_name = str(setup_id)
    checkpoints_path = os.path.join(
        "/".join(args.setups_file.split("/")[:-1]),
        "checkpoints", cp_name)
    if int(setup_id) < 10000:
        cp_name = params
        checkpoints_path = "/".join(cp_name.split("/")[:-1])
    local_args = params_from_args(params)
    conv_filters = [int(num) for num in local_args.conv_filters.split("-")]

    if not hasattr(local_args, "ico_order"):
        local_args.ico_order = len(conv_filters) + 1

    encoder = SphericalHemiFusionEncoder(
        n_features, local_args.ico_order, local_args.latent_dim,
        fusion_level=local_args.fusion_level, conv_flts=conv_filters,
        activation=local_args.activation, batch_norm=local_args.batch_norm,
        conv_mode="DiNe",
        cachedir=os.path.join(args.outdir, "cached_ico_infos"))
    
    on_the_fly_transform = None

    scaling = local_args.standardize
    scalers = {mod: None for mod in modalities}
    if scaling:
        for modality in modalities:
            path_to_scaler = os.path.join(args.datadir, f"{modality}_scaler.save")
            scaler = joblib.load(path_to_scaler)
            scalers[modality] =  transforms.Compose([
                Reshape((1, -1)),
                scaler.transform,
                transforms.ToTensor(),
                torch.squeeze,
                Reshape(input_shape),
            ])

    on_the_fly_transform = dict()
    for modality in modalities:
        transformer = Transformer()
        if args.standardize:
            transformer.register(scalers[modality])
        if args.normalize:
            transformer.register(Normalize())
        on_the_fly_transform[modality] = transformer

    dataset = DataManager(
        dataset=args.data, datasetdir=args.datadir, modalities=modalities,
        stratify=["sex", "age", "site"], discretize=["age"],
        transform=transform, on_the_fly_transform=on_the_fly_transform,
        overwrite=False, test_size="defaults", **kwargs)
    dataset.create_val_from_test(
        val_size=0.5, stratify=["sex", "age", "site"], discretize=["age"])

    train_loader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=batch_size, num_workers=6,
        pin_memory=True, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        dataset["test"]["valid"], batch_size=batch_size, num_workers=6,
        pin_memory=True, shuffle=True)

    all_metrics = {}
    for name in evaluation_metrics.keys():
        all_metrics[name] = []
    path_to_metrics = os.path.join(checkpoints_path, "validation_metrics.json")

    if (local_args.ico_order != 5 or os.path.exists(path_to_metrics)
        or not os.path.exists(checkpoints_path)):
        continue

    epochs = []
    is_finished = False
    print(checkpoints_path)
    for file in tqdm(os.listdir(checkpoints_path)):
        full_path = os.path.join(checkpoints_path, file)
        if not (os.path.isfile(full_path) and file.endswith("pth") and "model" in file):
            continue
        checkpoint = torch.load(full_path)
        checkpoint = encoder_cp_from_model_cp(checkpoint)
        epoch = int(file.split(".pth")[0].split("_")[-1])
        if epoch == local_args.epochs:
            is_finished = True
        epochs.append(epoch)
        encoder.load_state_dict(checkpoint)
        model = encoder.to(device)
        model.eval()
        regressor = Ridge()
        latents = []
        ys = []
        for step, x in enumerate(train_loader):
            x, metadata, _ = x
            left_x = x["surface-lh"].float().to(device, non_blocking=True)
            right_x = x["surface-rh"].float().to(device, non_blocking=True)
            if args.to_predict in metadata.keys():
                y = metadata[args.to_predict]
            else:
                y = x["clinical"][:, index_to_predict]
            ys.append(y)
            with torch.cuda.amp.autocast():
                X = (left_x, right_x)
                latents.append(model(X).squeeze().detach().cpu().numpy())
        y = np.concatenate(ys)
        X = np.concatenate(latents)
        regressor.fit(X, y)

        valid_latents = []
        valid_ys = []
        valid_transformed_ys = []
        for step, x in enumerate(valid_loader):
            x, metadata, _ = x
            left_x = x["surface-lh"].float().to(device, non_blocking=True)
            right_x = x["surface-rh"].float().to(device, non_blocking=True)
            if args.to_predict in metadata.keys():
                y = metadata[args.to_predict]
            else:
                y = x["clinical"][:, index_to_predict]
            valid_ys.append(y)
            with torch.cuda.amp.autocast():
                X = (left_x, right_x)
                valid_latents.append(model(X).squeeze().detach().cpu().numpy())

        X_valid = np.concatenate(valid_latents)
        y_valid = np.concatenate(valid_ys)

        y_hat_valid = regressor.predict(X_valid)
        for name, metric in evaluation_metrics.items():
            all_metrics[name].append(metric(y_valid, y_hat_valid))
    
    if len(epochs) > 0:
        best_epoch_per_metric = {}
        best_value_per_metric = {}
        for name in evaluation_metrics.keys():
            sorted_indices = np.argsort(all_metrics[name])
            best_index = 0 if what_is_best[name] == "lower" else -1
            best_epoch = epochs[sorted_indices[best_index]]
            best_value = all_metrics[name][sorted_indices[best_index]]
            best_epoch_per_metric[name] = best_epoch
            best_value_per_metric[name] = best_value
        print(str(setup_id) + ",".join([f" best {name} : {value}"
              for name, value in best_value_per_metric.items()]))
    all_metrics["epochs"] = epochs
    if is_finished:
        setups = pd.read_table(args.setups_file)
        setups.loc[setups["id"] == setup_id, "best_epoch"] = (
            best_epoch_per_metric[final_metric])
        setups.to_csv(args.setups_file, index=False, sep="\t")
        with open(path_to_metrics, 'w') as fp:
            json.dump(all_metrics, fp)
