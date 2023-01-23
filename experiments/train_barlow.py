import argparse
import json
import os
import sys
import time
import joblib
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from scipy.stats import norm
import pandas as pd
import random

import torch
from torch import nn, optim
from torch.distributions import Normal
from torchvision.utils import make_grid
from torchvision.transforms import Compose, RandomApply, ToPILImage
from torchvision import transforms
from surfify.models import SphericalVAE, SphericalGVAE, HemiFusionDecoder, HemiFusionEncoder, SphericalHemiFusionEncoder
from surfify.losses import SphericalVAELoss
from surfify.utils import setup_logging, icosahedron, text2grid, grid2text, downsample_data, downsample
from brainboard import Board

from multimodaldatasets.datasets import DataManager, DataLoaderWithBatchAugmentation
from augmentations import Permute, RescaleAsImage, Normalize, GaussianBlur, PermuteBeetweenModalities, Bootstrapping, Reshape, Cutout, Transformer


parser = argparse.ArgumentParser(description="Train Barlow Twins")
parser.add_argument(
    "--data", default="hbn", choices=("hbn", "euaims", "hcp", "openbhb"),
    help="the input cohort name.")
parser.add_argument(
    "--datadir", metavar="DIR", help="data directory path.", required=True)
parser.add_argument(
    "--ico-order", default=6, type=int,
    help="the icosahedron order.")
parser.add_argument(
    "--conv", default="DiNe", choices=("DiNe", "RePa", "SpMa"),
    help="Wether or not to project on a 2d grid")
parser.add_argument(
    "--epochs", default=100, type=int, metavar="N",
    help="number of total epochs to run.")
parser.add_argument(
    "--start-epoch", default=1, type=int, metavar="N",
    help="epoch from where to start.")
parser.add_argument(
    "--batch-size", "-bs", default=128, type=int, metavar="N",
    help="mini-batch size.")
# parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
#                     help='base learning rate for weights')
# parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
#                     help='base learning rate for biases and batch norm parameters')
parser.add_argument('--learning-rate', "-lr", default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument(
    "--weight-decay", default=1e-6, type=float, metavar="W",
    help="weight decay.")
parser.add_argument(
    "--conv-filters", default="64-128-128-256-256", type=str, metavar="F",
    help="convolutional filters at each layer.")
parser.add_argument(
    "--projector", default="256-512-512", type=str, metavar="F",
    help="convolutional filters at each layer.")
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
    "--gaussian-blur-augment", "-gba", action="store_true",
    help="optionnally uses gaussian blur augment.")
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

args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
args.ngpus_per_node = torch.cuda.device_count()
args.conv_filters = [int(item) for item in args.conv_filters.split("-")]


# Prepare process
setup_logging(level="info", logfile=None)
checkpoint_dir = os.path.join(args.outdir, "pretrain_barlow", "checkpoints")
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
stats_file = open(os.path.join(checkpoint_dir, "stats.txt"), "a", buffering=1)
print(" ".join(sys.argv))
print(" ".join(sys.argv), file=stats_file)

# Load the input cortical data
modalities = ["surface-lh", "surface-rh"]
metrics = ["thickness", "curv", "sulc"]
use_grid = args.conv == "SpMa"
transform = None
on_the_fly_transform = None
batch_transforms = None
batch_transforms_valid = None
on_the_fly_inter_transform = None
overwrite = False

grid_size = 192
limits_per_metric = {"curv": [-5, 5]}
def clip_values(textures, channel_dim=-1):
    metric_idx_to_clip = {key: metrics.index(key) for key in limits_per_metric}
    for key, idx in metric_idx_to_clip.items():
        clipped_values = np.expand_dims(np.clip(np.take(textures, idx, axis=channel_dim), *limits_per_metric[key]), axis=channel_dim)
        where = np.ones([1] * len(textures.shape), dtype=np.int64) * idx
        np.put_along_axis(textures, where, clipped_values, axis=channel_dim)
    return textures

def transform_to_grid_wrapper(order=args.ico_order, grid_size=grid_size,
                              channel_dim=1, new_channel_dim=-1, standard_ico=False):

    vertices, triangles = icosahedron(order, standard_ico=standard_ico)
    new_channel_dim = new_channel_dim if new_channel_dim > 0 else 4 + new_channel_dim
    def textures2grid(textures):
        new_textures = []
        for texture in textures:
            new_texture = []
            for channel_idx in range(texture.shape[channel_dim]):
                new_texture.append(text2grid(vertices, np.take(texture, channel_idx, axis=channel_dim)))
            new_textures.append(new_texture)
        new_textures = np.asarray(new_textures)
        new_order = list(range(len(new_textures.shape)))
        new_order.remove(1)
        new_order.insert(new_channel_dim + 1, 1)
        return new_textures.transpose(new_order)
    return textures2grid

def composed_transform(textures):
    transform_to_grid = transform_to_grid_wrapper()
    return clip_values(transform_to_grid(textures))
if use_grid:
    transform = {
        "surface-lh": composed_transform,#transform_to_grid_wrapper(),
        "surface-rh": composed_transform,#transform_to_grid_wrapper(),
    }
else:
    order = 7
    ico_verts, _ = icosahedron(order)
    down_indices = []
    for low_order in range(order - 1, args.ico_order - 1, -1):
        low_ico_verts, _ = icosahedron(low_order)
        down_indices.append(downsample(ico_verts, low_ico_verts))
        ico_verts = low_ico_verts
    def transform(x):
        downsampled_data = downsample_data(x, 7 - args.ico_order, down_indices)
        return np.swapaxes(downsampled_data, 1, 2)

input_shape = ((grid_size, grid_size, len(metrics)) if use_grid else
               (len(icosahedron(7)), len(metrics)))

kwargs = {
    "surface-rh": {"metrics": metrics},
    "surface-lh": {"metrics": metrics},
    # "clinical": {"z_score": False}
}

scalers = {mod: None for mod in modalities}
if args.batch_augment > 0 or args.standardize:
    original_dataset = DataManager(
        dataset=args.data, datasetdir=args.datadir,
        stratify_on=["sex", "age"], discretize=["age"],
        modalities=modalities, transform=transform,
        overwrite=overwrite, on_the_fly_transform=None,
        on_the_fly_inter_transform=None,
        test_size=None, **kwargs)

    loader = torch.utils.data.DataLoader(
        original_dataset["train"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        original_dataset["test"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
        shuffle=True)
    if args.batch_augment > 0:
        groups = {}
        groups_valid = {}
        batch_transforms = {}
        batch_transforms_valid = {}
    for modality in modalities:
        path_to_scaler = os.path.join(args.datadir, f"{modality}_scaler.save")
        if not os.path.exists(path_to_scaler) or args.batch_augment > 0 or overwrite:
            regressor = KNeighborsRegressor(n_neighbors=30)
            X, Y = [], []
            X_valid, Y_valid = [], []
            shape = None
            for x in loader:
                x, metadata, _ = x
                X += x[modality].view((len(x[modality]), -1)).tolist()
                Y += metadata["age"].tolist()
    
            for x in valid_loader:
                x, metadata, _ = x
                X_valid += x[modality].view((len(x[modality]), -1)).tolist()
                Y_valid += metadata["age"].tolist()
            
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

            print("Reducting")
            reductor = PCA(20)
            X = reductor.fit_transform(X)
            X_valid = reductor.transform(X_valid)
            print("Reducted")

            regressor.fit(X, Y)
            print(mean_absolute_error(Y, regressor.predict(X)))
            print(mean_absolute_error(Y_valid, regressor.predict(X_valid)))
            print(r2_score(Y, regressor.predict(X)))
            print(r2_score(Y_valid, regressor.predict(X_valid)))
            _, neigh_idx = regressor.kneighbors(X)
            _, neigh_idx_valid = regressor.kneighbors(X_valid)
            groups[modality] = neigh_idx
            groups_valid[modality] = neigh_idx_valid
            print("Groups built.")
                
            batch_transforms[modality] = Bootstrapping(
                p=(1, 0.1), p_corrupt=args.batch_augment,
                groups=groups[modality])
            batch_transforms_valid[modality] = Bootstrapping(
                p=(1, 0.1), p_corrupt=args.batch_augment,
                groups=groups_valid[modality])

class Transform:
    def __init__(self, normalize=False, scaler=None):
        self.transform, self.transform_prime = [], []

        if scaler is not None:
            self.transform.append(scaler)
            self.transform_prime.append(scaler)

        self.transform += [
            Permute((2, 0, 1)),
        ]
        self.transform_prime += [
            Permute((2, 0, 1)),
        ]

        if normalize:
            self.transform += [
                Normalize()
            ]
            self.transform_prime += [
                Normalize()
            ]
        if args.gaussian_blur_augment:
            self.transform += [
                RescaleAsImage(metrics),
                ToPILImage(),
                GaussianBlur(p=1.0),
                transforms.ToTensor(),
            ]
            self.transform_prime += [
                RescaleAsImage(metrics),
                ToPILImage(),
                GaussianBlur(p=0.1),
                transforms.ToTensor(),
            ]

        if args.cutout:
            self.transform += [
                Cutout(patch_size=np.ceil(np.array(input_shape)/4), p=1)
            ]

            self.transform_prime += [
                Cutout(patch_size=np.ceil(np.array(input_shape)/4), p=0.5)
            ]

        
        self.transform = transforms.Compose(self.transform)
        self.transform_prime = transforms.Compose(self.transform_prime)

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

normalize = args.normalize
if args.inter_modal_augment > 0 or args.batch_augment > 0:
    normalize = False

on_the_fly_transform = dict()
for modality in modalities:
    transformer = Transformer(["hard", "soft"])
    if args.standardize:
        transformer.register(scalers[modality])
    transformer.register(Permute((2, 0, 1)))
    if args.normalize:
        transformer.register(Normalize())
    if args.gaussian_blur_augment:
        transformer.register(RescaleAsImage(metrics))
        transformer.register(ToPILImage)
        transformer.register(GaussianBlur(), pipeline="hard")
        transformer.register(GaussianBlur(), probability=0.1, pipeline="soft")
        transformer.register(transforms.ToTensor())
    if args.cutout:
        transform = Cutout(patch_size=np.ceil(np.array(input_shape)/4))
        transformer.register(transform, pipeline="hard")
        transformer.register(transform, probability=0.5, pipeline="soft")
    on_the_fly_transform[modality] = transformer

# on_the_fly_transform = {
#     "surface-lh": Transform(normalize, scaler=scalers["surface-lh"]),
#     "surface-rh": Transform(normalize, scaler=scalers["surface-rh"])
# }

if args.inter_modal_augment > 0:
    normalizer = Normalize() if args.batch_augment == 0 else None
    on_the_fly_inter_transform = PermuteBeetweenModalities(
        (1, 0.1), args.inter_modal_augment, ("surface-lh", "surface-rh"),
        normalizer)


dataset = DataManager(dataset=args.data, datasetdir=args.datadir,
                      stratify_on=["sex", "age"], discretize=["age"],
                      modalities=modalities, transform=transform,
                      overwrite=False, on_the_fly_transform=on_the_fly_transform,
                      on_the_fly_inter_transform=on_the_fly_inter_transform,
                      test_size=None, **kwargs)

loader = DataLoaderWithBatchAugmentation(batch_transforms,
    dataset["train"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    shuffle=True)
valid_loader = DataLoaderWithBatchAugmentation(batch_transforms_valid,
    dataset["test"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    shuffle=True)

print(len(loader))
print(len(valid_loader))

activation = "ReLU"
n_features = len(metrics)

class SelectNthDim(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x[self.dim]

if use_grid:
    encoder = HemiFusionEncoder(n_features, grid_size, args.latent_dim,
                                fusion_level=args.fusion_level,
                                conv_flts=args.conv_filters,
                                activation=activation,
                                batch_norm=args.batch_norm,
                                return_dist=False)
    backbone = nn.Sequential(encoder, SelectNthDim(0))
else:
    backbone = SphericalHemiFusionEncoder(
        n_features, args.ico_order, args.latent_dim, fusion_level=args.fusion_level,
        conv_flts=args.conv_filters, activation=activation,
        batch_norm=args.batch_norm, conv_mode=args.conv,
        cachedir=os.path.join(args.outdir, "cached_ico_infos"))


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = 

        # projector
        sizes = [args.latent_dim] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss

model = BarlowTwins(args).to(device)


optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
if args.reduce_lr:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

run_name = ("deepint_barlow_{}_surf_{}_features_fusion_{}_act_{}_bn_{}_conv_{}"
    "_latent_{}_wd_{}_{}_epochs_lr_{}_bs_{}_ba_{}_ima_{}_gba_{}_cutout_{}"
    "_normalize_{}_standardize_{}").format(
        args.data, n_features, args.fusion_level, activation,
        args.batch_norm, "-".join([str(s) for s in args.conv_filters]), args.latent_dim,
        args.weight_decay, args.epochs, args.learning_rate, args.batch_size,
        args.batch_augment, args.inter_modal_augment, args.gaussian_blur_augment,
        args.cutout, args.normalize, args.standardize)

checkpoint_dir = os.path.join(checkpoint_dir, run_name)
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{}.pth")

use_board = False
if args.epochs > 0 and use_board:
    board = Board(env=run_name)

print(model)
print("Number of trainable parameters : ",
        sum(p.numel() for p in model.parameters() if p.requires_grad))
losses = []
valid_losses = []
start_epoch = args.start_epoch
if args.start_epoch > 1:
    checkpoint = torch.load(checkpoint_path.format(args.start_epoch), map_location=f"{device}:0")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    losses.append(checkpoint["loss"])
    valid_losses.append(checkpoint["valid_loss"])
    start_epoch += 1

threshold_valid_loss = 1000
start_time = time.time()
scaler = torch.cuda.amp.GradScaler()
best_saved_epoch = args.start_epoch
best_average_valid_loss = valid_losses[0] if args.start_epoch > 1 else 100000
for epoch in range(start_epoch, args.epochs + 1):
    stats = dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'],
                 loss=0, valid_loss=0)
    for step, data in enumerate(loader, start=epoch * len(loader)):
        data, _, _ = data
        y1_lh, y2_lh = data["surface-lh"]
        y1_rh, y2_rh = data["surface-rh"]
        y1_lh = y1_lh.float().to(device)
        y2_lh = y2_lh.float().to(device)
        y1_rh = y1_rh.float().to(device)
        y2_rh = y2_rh.float().to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = model.forward((y1_lh, y1_rh), (y2_lh, y2_rh))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        
        stats["loss"] += loss.item()
        stats["time"] = int(time.time() - start_time)
        stats["step"] = step
    mean_loss = stats["loss"] / len(loader) if not np.isinf(stats["loss"]) else losses[epoch-args.start_epoch-1]

    if use_board:
        board.update_plot("training loss", epoch, mean_loss)
    losses.append(mean_loss)

    model.eval()
    with torch.no_grad():
        for step, data in enumerate(valid_loader, start=epoch * len(valid_loader)):
            data, _, _ = data
            y1_lh, y2_lh = data["surface-lh"]
            y1_rh, y2_rh = data["surface-rh"]
            y1_lh = y1_lh.float().to(device)
            y2_lh = y2_lh.float().to(device)
            y1_rh = y1_rh.float().to(device)
            y2_rh = y2_rh.float().to(device)
            with torch.cuda.amp.autocast():
                loss = model.forward((y1_lh, y1_rh), (y2_lh, y2_rh))
            
            stats["valid_loss"] += loss.item()
            stats["time"] = int(time.time() - start_time)
    mean_loss = stats["valid_loss"] / len(valid_loader) if (stats["valid_loss"] / len(valid_loader)) < threshold_valid_loss or epoch == 1 else valid_losses[epoch-args.start_epoch-1]
    if use_board:
        board.update_plot("validation loss", epoch, mean_loss)
    valid_losses.append(mean_loss)

    model.train()
    if epoch % args.print_freq == 0:
        stats["loss"] /= len(loader)
        stats["valid_loss"] /= len(valid_loader)
        print(json.dumps(stats))
        print(json.dumps(stats), file=stats_file)

    if epoch % args.save_freq == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": losses[epoch-args.start_epoch],
            "valid_loss": mean_loss,
            }, checkpoint_path.format(epoch))
        if epoch != args.start_epoch:
            idx_epoch = epoch-args.start_epoch
            last_average_saved_valid_losses = np.mean(valid_losses[(idx_epoch - args.save_freq + 1):idx_epoch + 1])
            if last_average_saved_valid_losses < best_average_valid_loss:
                best_saved_epoch = epoch
        # torch.save(model.backbone[0].state_dict(),
        #     os.path.join(checkpoint_dir, "encoder_epoch_{}.pth".format(epoch)))
    if args.reduce_lr:
        scheduler.step()

plt.plot(range(args.start_epoch, args.epochs + 1), losses, label="training")
plt.plot(range(args.start_epoch, args.epochs + 1), valid_losses, label="validation")
plt.legend()
plt.title("Loss during training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(checkpoint_dir, "losses.pdf"))

torch.save(model.backbone[0].state_dict(),
            os.path.join(checkpoint_dir, "encoder.pth"))

if not os.path.exists(os.path.join(args.outdir, "pretrain_barlow", "setups.tsv")):
    setups = pd.DataFrame.from_dict(dict(id=[], path=[], epoch=[]))
    max_id = -1
else:
    setups = pd.read_table(os.path.join(args.outdir, "pretrain_barlow", "setups.tsv"))
    max_id = setups["id"].max()
setups = pd.concat([
    setups,
    pd.DataFrame({
        "id": [max_id + 1],
        "path": [os.path.join(checkpoint_dir, "encoder.pth")],
        "epoch": [best_saved_epoch]})],
    ignore_index=True)
setups.to_csv(os.path.join(args.outdir, "pretrain_barlow", "setups.tsv"),
              index=False, sep="\t")
