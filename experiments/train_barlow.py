import argparse
import json
import os
import sys
import time
import shutil
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, LogisticRegression
import pandas as pd
from neurocombat_sklearn import CombatModel

import torch
from torch import nn, optim
from torchvision import transforms
from surfify.models import SphericalHemiFusionEncoder
from surfify.augmentation import SphericalRandomCut, SphericalBlur, SphericalNoise
from surfify.utils import setup_logging, icosahedron, downsample_data, downsample, min_depth_to_get_n_neighbors
from brainboard import Board

from multimodaldatasets.datasets import DataManager, DataLoaderWithBatchAugmentation
from augmentations import Normalize, PermuteBeetweenModalities, Bootstrapping, Reshape, Transformer, SwitchModalities
from models import BarlowTwins, yAwareSimCLR, simCLR


parser = argparse.ArgumentParser(description="Train Barlow Twins")
parser.add_argument(
    "--data", default="hbn", choices=("hbn", "euaims", "hcp", "openbhb"),
    help="the input cohort name.")
parser.add_argument(
    "--datadir", metavar="DIR", help="data directory path.", required=True)
parser.add_argument(
    "--ico-order", default=5, type=int,
    help="the icosahedron order.")
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N",
    help="number of total epochs to run.")
parser.add_argument(
    "--start-epoch", default=1, type=int, metavar="N",
    help="epoch from where to start.")
parser.add_argument(
    "--batch-size", "-bs", default=128, type=int, metavar="N",
    help="mini-batch size.")
parser.add_argument(
    '--learning-rate', "-lr", default=1e-3, type=float,
    help='learning rate')
parser.add_argument(
    '--loss-param', default=0.0051, type=float, metavar='L',
    help='weight on off-diagonal terms')
parser.add_argument(
    "--weight-decay", default=1e-6, type=float, metavar="W",
    help="weight decay.")
parser.add_argument(
    "--conv-filters", default="128-128-256-256", type=str, metavar="F",
    help="convolutional filters at each layer.")
parser.add_argument(
    "--projector", default="256-512-512", type=str, metavar="F",
    help="projector linear layers.")
parser.add_argument(
    "--fusion-level", default=1, type=int, metavar="W",
    help="the fusion level in the convolutional encoder and decoder.")
parser.add_argument(
    "--batch-norm", "-bn", action="store_true",
    help="optionnally uses batch normalization.")
parser.add_argument(
    "--latent-dim", default=64, type=int, metavar="N",
    help="the number of latent dimension.")
parser.add_argument(
    "--outdir", metavar="DIR", help="output directory path.", required=True)
parser.add_argument(
    "--print-freq", default=1, type=int,
    help="printing frequence (as number of epochs) during training.")
parser.add_argument(
    "--save-freq", default=10, type=int,
    help="saving frequence (as number of epochs) during training.")
parser.add_argument(
    "--batch-augment", "-ba", default=0.0, type=float,
    help="optionnally uses batch augmentation.")
parser.add_argument(
    "--inter-modal-augment", "-ima", default=0.0, type=float,
    help="optionnally uses inter modality augment.")
parser.add_argument(
    "--blur", action="store_true",
    help="optionnally uses gaussian blur augment.")
parser.add_argument(
    "--noise", action="store_true",
    help="optionnally uses noise augment.")
parser.add_argument(
    "--cutout", action="store_true",
    help="optionnally uses cut out augment.")
parser.add_argument(
    "--normalize", action="store_true",
    help="optionnally normalizes input.")
parser.add_argument(
    "--standardize", action="store_true",
    help="optionnally standardize input with statistics computed across"
         "the train dataset.")
parser.add_argument(
    "--reduce-lr", action="store_true",
    help="optionnally reduces the learning rate during training.")
parser.add_argument(
    "--sigma", default=0.0, type=float,
    help="y-aware sigma parameter.")
parser.add_argument(
    "--algo", default="barlow", choices=("barlow", "simCLR"),
    help="the self-supervised algo.")
parser.add_argument(
    "--flip", action="store_true",
    help="optionnally uses flip augment.")

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
args.ngpus_per_node = torch.cuda.device_count()
args.conv_filters = [int(item) for item in args.conv_filters.split("-")]
args.conv = "DiNe"

# Load the input cortical data
modalities = ["surface-lh", "surface-rh"]
metrics = ["thickness", "curv", "sulc"]
transform = None
on_the_fly_transform = None
batch_transforms = None
batch_transforms_valid = None
on_the_fly_inter_transform = None
overwrite = False
n_features = len(metrics)
activation = "ReLU"

params = ("pretrain_{}_on_{}_surf_order_{}_with_{}_features_fusion_{}_act_{}"
    "_bn_{}_conv_{}_latent_{}_wd_{}_{}_epochs_lr_{}_reduced_{}_bs_{}_ba_{}_ima"
    "_{}_blur_{}_noise_{}_cutout_{}_normalize_{}_standardize_{}_loss_param_{}_"
    "sigma_{}").format(
        args.algo, args.data, args.ico_order, n_features, args.fusion_level,
        activation, args.batch_norm, "-".join([str(s) for s in args.conv_filters]),
        args.latent_dim, args.weight_decay, args.epochs, args.learning_rate,
        args.reduce_lr, args.batch_size, args.batch_augment,
        args.inter_modal_augment, args.blur, args.noise, args.cutout,
        args.normalize, args.standardize, args.loss_param, args.sigma)

# Prepare process
setup_logging(level="info", logfile=None)
checkpoint_dir = os.path.join(args.outdir, "pretrain", "checkpoints")
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
stats_file = open(os.path.join(checkpoint_dir, "stats.txt"), "a", buffering=1)
print(" ".join(sys.argv))
print(" ".join(sys.argv), file=stats_file)

setups = pd.read_table(os.path.join(args.outdir, "pretrain", "setups.tsv"))
run_id = int(time.time())

def same_params_but_epochs(args_str):
    epochs = args_str.split("_epochs")[0].split("_")[-1]
    return (params == args_str.replace(
        f"{epochs}_epochs", f"{args.epochs}_epochs"))

if args.start_epoch > 1:
    old_run_id = setups.loc[setups.args == params, "id"]
    other_run_id = setups.loc[setups.args.apply(same_params_but_epochs), "id"]
    if len(old_run_id) == 0 and len(other_run_id) > 0:
        other_run_id = other_run_id.values[0]
        print(other_run_id)
        old_path = os.path.join(checkpoint_dir, str(other_run_id))
        new_path = os.path.join(checkpoint_dir, str(run_id))
        os.makedirs(new_path)
        for file_name in os.listdir(old_path):
            old_file = os.path.join(old_path, file_name)
            new_file = os.path.join(new_path, file_name)
            if os.path.isfile(old_file):
                shutil.copy(old_file, new_file)
    else:
        run_id = old_run_id.item()
else:
    setups = pd.concat([
        setups,
        pd.DataFrame({
            "id": [run_id],
            "args": [params],
            "best_epoch": [0],
            "best_param": [1],
            "best_value": [1000]})],
        ignore_index=True)
    setups.to_csv(os.path.join(args.outdir, "pretrain", "setups.tsv"),
        index=False, sep="\t")
print(run_id)


order = 7
ico_verts, ico_tri = icosahedron(order)
down_indices = []
for low_order in range(order - 1, args.ico_order - 1, -1):
    low_ico_verts, low_ico_tri = icosahedron(low_order)
    down_indices.append(downsample(ico_verts, low_ico_verts))
    ico_verts = low_ico_verts
    ico_tri = low_ico_tri
def transform(x):
    downsampled_data = downsample_data(x, 7 - args.ico_order, down_indices)
    return np.swapaxes(downsampled_data, 1, 2)

input_shape = (len(metrics), len(ico_verts))

backbone = SphericalHemiFusionEncoder(
    n_features, args.ico_order, args.latent_dim, fusion_level=args.fusion_level,
    conv_flts=args.conv_filters, activation=activation,
    batch_norm=args.batch_norm, conv_mode=args.conv,
    cachedir=os.path.join(args.outdir, "cached_ico_infos"))

kwargs = {
    "surface-rh": {"metrics": metrics},
    "surface-lh": {"metrics": metrics},
    # "clinical": {"z_score": False}
}

scalers = {mod: None for mod in modalities}
if args.batch_augment > 0 or args.standardize:
    original_dataset = DataManager(
        dataset=args.data, datasetdir=args.datadir,
        stratify_on=["sex", "age", "site"], discretize=["age"],
        modalities=modalities, transform=transform,
        overwrite=overwrite, on_the_fly_transform=None,
        on_the_fly_inter_transform=None,
        test_size=None, **kwargs)

    loader = torch.utils.data.DataLoader(
        original_dataset["train"], batch_size=args.batch_size, num_workers=6,
        pin_memory=True, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        original_dataset["test"], batch_size=args.batch_size, num_workers=6,
        pin_memory=True, shuffle=True)
    if args.batch_augment > 0:
        groups = {}
        groups_valid = {}
        batch_transforms = {}
        batch_transforms_valid = {}
    for modality in modalities:
        path_to_scaler = os.path.join(args.datadir, f"{modality}_scaler.save")
        if not os.path.exists(path_to_scaler) or args.batch_augment > 0 or overwrite:
            regressor = KNeighborsRegressor(n_neighbors=30)
            residualizer = CombatModel()
            check_site_biasor = LogisticRegression()
            X, Y, sites, sex = [], [], [], []
            X_valid, Y_valid, sites_valid, sex_valid = [], [], [], []
            shape = None
            for x in loader:
                x, metadata, _ = x
                X += x[modality].view((len(x[modality]), -1)).tolist()
                Y += metadata["age"].tolist()
                sites += metadata["site"].tolist()
                sex += metadata["sex"]
    
            for x in valid_loader:
                x, metadata, _ = x
                X_valid += x[modality].view((len(x[modality]), -1)).tolist()
                Y_valid += metadata["age"].tolist()
                sites_valid += metadata["site"].tolist()
                sex_valid += metadata["sex"]
            
            X = np.asarray(X)
            X_valid = np.asarray(X_valid)
            Y = np.asarray(Y)
            Y_valid = np.asarray(Y_valid)
            sites = np.asarray(sites)
            sites_valid = np.asarray(sites_valid)
            sex = np.asarray(sex)
            sex_valid = np.asarray(sex_valid)
            
            if not os.path.exists(path_to_scaler) or overwrite:
                print("Fit scaler")
                scaler = StandardScaler()
                scaler.fit(X)
                joblib.dump(scaler, path_to_scaler)

        scaler = joblib.load(path_to_scaler)
        
        if args.standardize:
            scalers[modality] =  transforms.Compose([
                Reshape((1, -1)),
                scaler.transform,
                transforms.ToTensor(),
                torch.squeeze,
                Reshape(input_shape),
            ])
        if args.batch_augment > 0:
            print(f"Initializing KNN for {modality}")
            X = scaler.transform(X)
            X_valid = scaler.transform(X_valid)
            print("Scaled")

            print("Residualizing")
            zero_mask = (X == 0).sum(0) == len(X)
            X[:, ~zero_mask] = residualizer.fit_transform(
                X[:, ~zero_mask], sites[:, None], sex[:, None], Y[:, None])
            X_valid[:, ~zero_mask] = residualizer.transform(
                X_valid[:, ~zero_mask], sites_valid[:, None],
                sex_valid[:, None], Y_valid[:, None])
            print("Residualized")

            print("Reducting")
            reductor = PCA(20)
            X = reductor.fit_transform(X)
            X_valid = reductor.transform(X_valid)
            print("Reducted")

            regressor.fit(X, Y)
            # check_site_biasor.fit(X, sites)
            # sites_hat = check_site_biasor.predict(X)
            # sites_valid_hat = check_site_biasor.predict(X_valid)

            # print(balanced_accuracy_score(sites, sites_hat))
            # print(balanced_accuracy_score(sites_valid, sites_valid_hat))
            # print(mean_absolute_error(Y, regressor.predict(X)))
            # print(mean_absolute_error(Y_valid, regressor.predict(X_valid)))
            # print(r2_score(Y, regressor.predict(X)))
            # print(r2_score(Y_valid, regressor.predict(X_valid)))
            _, neigh_idx = regressor.kneighbors(X)
            _, neigh_idx_valid = regressor.kneighbors(X_valid)
            groups[modality] = neigh_idx
            groups_valid[modality] = neigh_idx_valid
            print("Groups built.")
            
            probabilities = (0.5, 0.5)#(1, 0.1) if args.algo == "barlow" else (0.5, 0.5)
            batch_transforms[modality] = Bootstrapping(
                p=probabilities, p_corrupt=args.batch_augment,
                groups=groups[modality])
            batch_transforms_valid[modality] = Bootstrapping(
                p=probabilities, p_corrupt=args.batch_augment,
                groups=groups_valid[modality])


normalize = args.normalize
if args.inter_modal_augment > 0 or args.batch_augment > 0:
    normalize = False

on_the_fly_transform = dict()
for modality in modalities:
    transformer = Transformer(["hard", "soft"])
    if args.standardize:
        transformer.register(scalers[modality])
    if normalize:
        transformer.register(Normalize())
    if args.blur:
        ico = backbone.ico[args.ico_order]
        trf = SphericalBlur(
            ico.vertices, ico.triangles, None,
            sigma=(0.1, 2),
            cachedir=os.path.join(args.outdir, "cached_ico_infos"))
        if args.algo == "barlow":
            transformer.register(trf, pipeline="hard")
            transformer.register(trf, probability=0.1, pipeline="soft")
        else:
            transformer.register(trf, probability=0.5)
    if args.noise:
        trf = SphericalNoise(sigma=(0.1, 2))
        if args.algo == "barlow":
            transformer.register(trf, pipeline="hard")
            transformer.register(trf, probability=0.1, pipeline="soft")
        else:
            transformer.register(trf, probability=0.5)
    if args.cutout:
        ico = backbone.ico[args.ico_order]
        t = time.time()
        path_size = min_depth_to_get_n_neighbors(np.ceil(len(ico.vertices) / 4))
        trf = SphericalRandomCut(
            ico.vertices, ico.triangles, None,
            patch_size=path_size,
            cachedir=os.path.join(args.outdir, "cached_ico_infos"))
        # print(time.time() - t)
        if args.algo == "barlow":
            transformer.register(trf, pipeline="hard")
            transformer.register(trf, probability=0.5, pipeline="soft")
        else:
            transformer.register(trf, probability=0.5)
    on_the_fly_transform[modality] = transformer

if args.inter_modal_augment > 0:
    normalizer = Normalize() if args.batch_augment == 0 and normalize else None
    probabilities = (0.5, 0.5)#(1, 0.1) if args.algo == "barlow" else (0.5, 0.5)
    on_the_fly_inter_transform = PermuteBeetweenModalities(
        probabilities, args.inter_modal_augment, ("surface-lh", "surface-rh"),
        normalizer)


dataset = DataManager(
    dataset=args.data, datasetdir=args.datadir,
    stratify_on=["sex", "age", "site"], discretize=["age"],
    modalities=modalities, transform=transform,
    overwrite=False, on_the_fly_transform=on_the_fly_transform,
    on_the_fly_inter_transform=on_the_fly_inter_transform,
    test_size=None, **kwargs)


valid_on_the_fly_transform = dict()
for modality in modalities:
    transformer = Transformer()
    if args.standardize:
        transformer.register(scalers[modality])
    if args.normalize:
        transformer.register(Normalize())
    valid_on_the_fly_transform[modality] = transformer

dataset_valid = DataManager(
    dataset=args.data, datasetdir=args.datadir,
    stratify=["sex", "age", "site"], discretize=["age"],
    modalities=modalities, transform=transform,
    overwrite=False, on_the_fly_transform=valid_on_the_fly_transform,
    test_size=None, **kwargs)
dataset_valid.create_val_from_test(
    val_size=0.5, stratify=["sex", "age", "site"], discretize=["age"])

loader = DataLoaderWithBatchAugmentation(
    batch_transforms, dataset["train"], batch_size=args.batch_size,
    num_workers=6, pin_memory=True, shuffle=True)
valid_loader = DataLoaderWithBatchAugmentation(
    batch_transforms_valid, dataset["test"], batch_size=args.batch_size,
    num_workers=6, pin_memory=True, shuffle=True)
train_no_augment_loader = torch.utils.data.DataLoader(
    dataset_valid["train"], batch_size=args.batch_size, num_workers=6,
    pin_memory=True, shuffle=True)
valid_no_augment_loader = torch.utils.data.DataLoader(
    dataset_valid["test"]["valid"], batch_size=args.batch_size, num_workers=6,
    pin_memory=True, shuffle=True)

# print(len(loader))
# print(len(valid_loader))

if args.algo == "barlow":
    model = BarlowTwins(args, backbone).to(device)
elif args.sigma > 0:
    model = yAwareSimCLR(args, backbone, return_logits=True).to(device)
else:
    model = simCLR(args, backbone, return_logits=True).to(device)


optimizer = optim.Adam(
    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
if args.reduce_lr:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

checkpoint_dir = os.path.join(checkpoint_dir, str(run_id))
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{}.pth")

use_board = False
if args.epochs > 0 and use_board:
    board = Board(env=args)

eval_metrics = {"mae": mean_absolute_error}
metric_for_perf = "mae"
regressor_params = [0.01, 0.1, 1, 10, 100]
regressor = Ridge

print(model)
print("Number of trainable parameters : ",
        sum(p.numel() for p in model.parameters() if p.requires_grad))
losses = []
valid_losses = []
perfs = []
valid_perfs = []
start_epoch = args.start_epoch
if args.start_epoch > 1:
    checkpoint = torch.load(checkpoint_path.format(args.start_epoch), map_location=f"{device}:0")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    losses.append(checkpoint["loss"])
    valid_losses.append(checkpoint["valid_loss"])
    if "perf" in checkpoint.keys():
        perfs.append(checkpoint["perf"])
        valid_perfs.append(checkpoint["valid_perf"])
    start_epoch += 1

threshold_valid_loss = 1000
start_time = time.time()
scaler = torch.cuda.amp.GradScaler()
best_saved_epoch = args.start_epoch
best_valid_perf = 10000 if len(valid_perfs) == 0 else valid_perfs[0]
for epoch in range(start_epoch, args.epochs + 1):
    stats = dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'],
                 loss=0, valid_loss=0)
    for step, data in enumerate(loader, start=epoch * len(loader)):
        data, metadata, _ = data
        y1_lh, y2_lh = data["surface-lh"]
        y1_rh, y2_rh = data["surface-rh"]
        y1_lh = y1_lh.float().to(device)
        y2_lh = y2_lh.float().to(device)
        y1_rh = y1_rh.float().to(device)
        y2_rh = y2_rh.float().to(device)
        labels = metadata["age"].float().to(device)
        forwarded = [(y1_lh, y1_rh), (y2_lh, y2_rh)]
        if args.algo == "simCLR" and args.sigma > 0:
            forwarded.append(labels)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = model.forward(*forwarded)
        if args.algo == "simCLR":
            loss = loss[0]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        
        stats["loss"] += loss.item()
        stats["step"] = step
    mean_loss = stats["loss"] / len(dataset["train"]) if not np.isinf(stats["loss"]) else losses[epoch-args.start_epoch-1]

    if use_board:
        board.update_plot("training loss", epoch, mean_loss)
    losses.append(mean_loss)

    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            data, metadata, _ = data
            y1_lh, y2_lh = data["surface-lh"]
            y1_rh, y2_rh = data["surface-rh"]
            y1_lh = y1_lh.float().to(device)
            y2_lh = y2_lh.float().to(device)
            y1_rh = y1_rh.float().to(device)
            y2_rh = y2_rh.float().to(device)
            labels = metadata["age"].float().to(device)
            forwarded = [(y1_lh, y1_rh), (y2_lh, y2_rh)]
            if args.algo == "simCLR" and args.sigma > 0:
                forwarded.append(labels)
            with torch.cuda.amp.autocast():
                loss = model.forward(*forwarded)
            
            if args.algo == "simCLR":
                loss, logits, target = loss
            
            stats["valid_loss"] += loss.item()

        all_representations = []
        all_labels = []
        for data in train_no_augment_loader:
            data, metadata, _ = data
            y1_lh = data["surface-lh"]
            y1_rh = data["surface-rh"]
            y1_lh = y1_lh.float().to(device)
            y1_rh = y1_rh.float().to(device)
            labels = metadata["age"].float().to(device)
            with torch.cuda.amp.autocast():
                representations = model.backbone.forward((y1_lh, y1_rh))
            
            all_labels.append(labels.detach().cpu().numpy())
            all_representations.append(representations.detach().cpu().numpy())

        all_representations = np.concatenate(all_representations)
        all_labels = np.concatenate(all_labels)
        regressors = [regressor(alpha=param) for param in regressor_params]
        for reg in regressors:
            reg.fit(all_representations, all_labels)
        preds = [
            reg.predict(all_representations) for reg in regressors]
        for name, metric in eval_metrics.items():
            best_metric = 1000
            for pred in preds:
                metric_value = metric(pred, all_labels)
                if metric_value < best_metric:
                    best_metric = metric_value
            stats[name] = best_metric

        all_representations = []
        all_labels = []
        for data in valid_no_augment_loader:
            data, metadata, _ = data
            y1_lh = data["surface-lh"]
            y1_rh = data["surface-rh"]
            y1_lh = y1_lh.float().to(device)
            y1_rh = y1_rh.float().to(device)
            labels = metadata["age"].float().to(device)
            with torch.cuda.amp.autocast():
                representations = model.backbone.forward((y1_lh, y1_rh))
            
            all_labels.append(labels.detach().cpu().numpy())
            all_representations.append(representations.detach().cpu().numpy())

        all_representations = np.concatenate(all_representations)
        all_labels = np.concatenate(all_labels)
        preds = [
            reg.predict(all_representations) for reg in regressors]
        for name, metric in eval_metrics.items():
            subname = "valid_" + name
            best_metric = 1000
            for param_idx, pred in enumerate(preds):
                metric_value = metric(pred, all_labels)
                if metric_value < best_metric:
                    best_metric = metric_value
                    best_param_idx = param_idx
            stats[subname] = best_metric
            
    stats["time"] = int(time.time() - start_time)
    mean_loss = (
        stats["valid_loss"] / len(dataset["test"]) if
            (stats["valid_loss"] / len(valid_loader)) < threshold_valid_loss
            or epoch == 1 else valid_losses[epoch-args.start_epoch-1])
    if use_board:
        board.update_plot("validation loss", epoch, mean_loss)
    valid_losses.append(mean_loss)
    perfs.append(stats[metric_for_perf])
    valid_perfs.append(stats["valid_" + metric_for_perf])

    model.train()
    if epoch % args.print_freq == 0:
        stats["loss"] /= len(dataset["train"])
        stats["valid_loss"] /= len(dataset["test"])
        for name, value in stats.items():
            if type(value) in [float, np.float32, np.float64]:
                stats[name] = round(value, 8)
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)

    if epoch % args.save_freq == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": losses[epoch-args.start_epoch],
            "valid_loss": mean_loss,
            "perf": perfs[epoch-args.start_epoch],
            "valid_perf": valid_perfs[epoch-args.start_epoch],
            }, checkpoint_path.format(epoch))
        if epoch != args.start_epoch:
            idx_epoch = epoch-args.start_epoch
            last_valid_perfs = valid_perfs[idx_epoch]
            if last_valid_perfs < best_valid_perf:
                best_saved_epoch = epoch
                best_valid_perf = last_valid_perfs
                setups = pd.read_table(
                    os.path.join(args.outdir, "pretrain", "setups.tsv"))
                setups.loc[setups["id"] == run_id,
                           ["best_epoch", "best_param", "best_value"]] = (
                    best_saved_epoch, regressor_params[best_param_idx],
                    best_valid_perf)
                setups.to_csv(os.path.join(args.outdir, "pretrain", "setups.tsv"),
                    index=False, sep="\t")
        # torch.save(model.backbone[0].state_dict(),
        #     os.path.join(checkpoint_dir, "encoder_epoch_{}.pth".format(epoch)))
    if args.reduce_lr:
        scheduler.step()

plt.plot(range(args.start_epoch, args.epochs + 1), losses,
         label="training")
plt.plot(range(args.start_epoch, args.epochs + 1), valid_losses,
         label="validation")
plt.legend()
plt.title("Loss during training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(
    checkpoint_dir, f"losses_from_{args.start_epoch}_to_{args.epochs}.pdf"))

plt.figure()
plt.plot(range(args.start_epoch, args.epochs + 1), perfs, label="training")
plt.plot(range(args.start_epoch, args.epochs + 1), valid_perfs,
         label="validation")
plt.legend()
plt.title("Performance during training")
plt.xlabel("Epoch")
plt.ylabel(metric_for_perf)
plt.savefig(os.path.join(
    checkpoint_dir, f"perfs_from_{args.start_epoch}_to_{args.epochs}.pdf"))

idx_epoch = epoch-args.start_epoch
last_average_valid_perfs = np.mean(valid_perfs[max((idx_epoch - args.save_freq + 1), 0):idx_epoch + 1])
if last_average_valid_perfs < best_valid_perf:
    setups = pd.read_table(os.path.join(args.outdir, "pretrain", "setups.tsv"))
    setups.loc[setups["id"] == run_id, "best_epoch"] = epoch
    setups.to_csv(os.path.join(args.outdir, "pretrain", "setups.tsv"),
        index=False, sep="\t")

module_to_save = model.backbone
torch.save(module_to_save.state_dict(),
           os.path.join(checkpoint_dir, "encoder.pth"))



