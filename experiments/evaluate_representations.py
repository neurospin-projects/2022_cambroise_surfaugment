import argparse
import json
import os
import sys
import time
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from scipy.stats import norm
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torch import nn, optim
from torch.distributions import Normal
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.transforms import Compose, RandomCrop, RandomRotation, RandomApply, RandomResizedCrop, InterpolationMode, ToPILImage
# from pl_bolts.models.autoencoders import VAE
from surfify.models import SphericalVAE, SphericalGVAE, HemiFusionDecoder, HemiFusionEncoder, SphericalHemiFusionEncoder
from surfify.losses import SphericalVAELoss
from surfify.utils import setup_logging, icosahedron, text2grid, grid2text, downsample_data, downsample
from surfify.augmentation import SphericalRandomCut, SphericalBlur

from brainboard import Board

from multimodaldatasets.datasets import DataManager, DataLoaderWithBatchAugmentation
from augmentations import Permute, Normalize, Reshape, Transformer, Cutout, GaussianBlur
import parse


# Get user parameters
parser = argparse.ArgumentParser(description="Spherical predictor")
parser.add_argument(
    "--data", default="hcp", choices=("hbn", "euaims", "hcp", "openbhb", "privatebhb"),
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
    "--outdir", metavar="DIR", help="output directory path.", required=True)
parser.add_argument(
    "--to-predict", default="age",
    help="the name of the variable to predict.")
parser.add_argument(
    "--method", default="regression", choices=("regression", "classification"),
    help="the prediction method.")
parser.add_argument(
    "--bin-step", default=1, type=int,
    help="the size of a bin when method is 'distribution'.")
parser.add_argument(
    "--sigma", default=1, type=float,
    help="the soft labels' gaussian's sigma when method is 'distribution'.")
parser.add_argument(
    "--pretrained-setup", default="None",
    help="the pretrained encoder's id.")
parser.add_argument(
    "--setups-file", default="None",
    help="the path to the file linking the setups to the pretrained encoder's path.")
parser.add_argument(
    "--evaluate", action="store_true",
    help="Uses the whole training set to train once, and evaluates on the hold-out test set."
)
args = parser.parse_args()

# Prepare process
setup_logging(level="info", logfile=None)
resdir = os.path.join(args.outdir, "evaluate_by_predicting_{}".format(args.to_predict))
if not os.path.isdir(resdir):
    os.makedirs(resdir)
stats_file = open(os.path.join(resdir, "stats.txt"), "a", buffering=1)
print(" ".join(sys.argv))
print(" ".join(sys.argv), file=stats_file)

# Load the input cortical data
modalities = ["surface-lh", "surface-rh"]
metrics = ["thickness", "curv", "sulc"]
use_grid = args.conv == "SpMa"
transform = None

input_size = 192
limits_per_metric = {"curv": [-5, 5]}
def clip_values(textures, channel_dim=-1):
    metric_idx_to_clip = {key: metrics.index(key) for key in limits_per_metric}
    for key, idx in metric_idx_to_clip.items():
        clipped_values = np.expand_dims(np.clip(np.take(textures, idx, axis=channel_dim), *limits_per_metric[key]), axis=channel_dim)
        where = np.ones([1] * len(textures.shape), dtype=np.int64) * idx
        np.put_along_axis(textures, where, clipped_values, axis=channel_dim)
    return textures


def transform_to_grid_wrapper(order=args.ico_order, grid_size=input_size, channel_dim=1,
                              new_channel_dim=-1, standard_ico=False):

    vertices, triangles = icosahedron(order, standard_ico=standard_ico)
    new_channel_dim = new_channel_dim if new_channel_dim > 0 else 4 + new_channel_dim
    def textures2grid(textures):
        new_textures = []
        for texture in textures:
            new_texture = text2grid(vertices, np.take(texture, 0, axis=channel_dim))[np.newaxis, :]
            for channel_idx in range(1, texture.shape[channel_dim]):
                new_texture = np.concatenate((
                    new_texture,
                    [text2grid(vertices, np.take(texture, channel_idx, axis=channel_dim))]))
            new_textures.append(new_texture)
        new_order = list(range(len(new_texture.shape) + 1))
        new_order.remove(1)
        new_order.insert(new_channel_dim + 1, 1)
        return np.asarray(new_textures).transpose(new_order)
    return textures2grid

def composed_transform(textures):
    transform_to_grid = transform_to_grid_wrapper()
    return clip_values(transform_to_grid(textures))

use_mlp = False
if use_mlp:
    use_grid = False
    input_size = 2 * len(metrics) * len(icosahedron(7)[0])

input_shape = ((input_size, input_size, len(metrics)) if use_grid else
               (len(metrics), len(icosahedron(args.ico_order)[0])))

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


def encoder_cp_from_barlow_cp(checkpoint):
    idx_to_start_from = 1
    name_to_check = "backbone"
    if use_grid:
        name_to_check = "backbone.0"
        idx_to_start_from = 2
    checkpoint = {".".join(key.split(".")[idx_to_start_from:]): value 
                for key, value in checkpoint["model_state_dict"].items() if name_to_check in key}
    return checkpoint

def params_from_path(path):
    format = (
        "deepint_barlow_{}_surf_{}_features_fusion_{}_act_{}_bn_{}_conv_{}"
        "_latent_{}_wd_{}_{}_epochs_lr_{}_bs_{}_ba_{}_ima_{}_gba_{}_cutout_{}"
        "_normalize_{}_standardize_{}")
    parsed = parse.parse(format, path.split("/")[-2])
    args_names = ["data_train", "n_features", "fusion_level", "activation",
        "batch_norm", "conv_filters", "latent_dim",
        "weight_decay", "epochs", "learning_rate", "batch_size",
        "batch_augment", "inter_modal_augment", "gaussian_blur_augment",
        "cutout", "normalize", "standardize"]
    for idx, value in enumerate(parsed.fixed):
        if value.isdigit():
            value = float(value)
            value = int(value) if int(value) == value else value
        elif value in ["True", "False"]:
            value = value == "True"
        setattr(args, args_names[idx], value)
    return args

checkpoint = None
if args.pretrained_setup != "None":
    assert args.setups_file != "None"
    setups = pd.read_table(args.setups_file)
    pretrained_path, epoch = setups[setups["id"] == int(args.pretrained_setup)][["path", "epoch"]].values[0]
    args = params_from_path(pretrained_path)
    max_epoch = int(pretrained_path.split("_epochs")[0].split("_")[-1])
    if epoch != max_epoch:
        pretrained_path = pretrained_path.replace("encoder.pth", "model_epoch_{}.pth".format(int(epoch)))
    checkpoint = torch.load(pretrained_path)
    if epoch != max_epoch:
        checkpoint = encoder_cp_from_barlow_cp(checkpoint)
else:
    args.n_features = len(metrics)
    args.fusion_level = 1
    args.activation = "ReLU"
    args.standardize = True
    args.normalize = True
    args.batch_norm = False
    args.conv_filters = "64-128-128-256-256"
    args.latent_dim = 64
    args.gaussian_blur_augment = False
    args.cutout = False
args.batch_size = 32

print(args)

on_the_fly_transform = None

scaling = args.standardize
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
    if use_grid:
        channels_to_switch = (2, 0, 1)
        transformer.register(Permute(channels_to_switch))
    if args.normalize:
        transformer.register(Normalize())
    on_the_fly_transform[modality] = transformer


# downsampler = wrapper_data_downsampler(args.outdir, to_order=args.ico_order)

kwargs = {
    "surface-rh": {"metrics": metrics},
    "surface-lh": {"metrics": metrics},
    "multiblock": {"test_size": 0.2}
}

if args.data in ["hbn", "euaims"]:
    kwargs["surface-lh"]["symetrized"] = True
    kwargs["surface-rh"]["symetrized"] = True

    modalities.append("clinical")

if args.data == "openbhb":
    kwargs["multiblock"]["test_size"] = None

n_folds = 5

dataset = DataManager(dataset=args.data, datasetdir=args.datadir,
                      modalities=modalities, validation=n_folds,
                      stratify_on=["sex", "age"], discretize=["age"],
                      transform=transform, on_the_fly_transform=on_the_fly_transform,
                      overwrite=False, **kwargs)


loader = torch.utils.data.DataLoader(
    dataset["train"]["all"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    shuffle=True)

class SoftBiner(object):
    def __init__(self, bin_step, sigma):
        super().__init__()
        self.sigma = sigma
        self.bin_step = bin_step
    
    def fit(self, x):
        self.bin_start = np.floor(x.min())
        self.bin_end = np.ceil(x.max())
        bin_length = self.bin_end - self.bin_start
        if not bin_length % self.bin_step == 0:
            print("bin's range should be divisible by bin_step!")
            return -1
        bin_number = int(bin_length / self.bin_step)
        self.bin_centers = self.bin_start + self.bin_step / 2 + self.bin_step * np.arange(bin_number)


    def transform(self, x):
        if x.ndim > 1:
            x = x.squeeze()
        if self.sigma == 0:
            i = np.floor((np.array(x) - self.bin_start) / self.bin_step)
            i = i.astype(int)
            return i
        elif self.sigma > 0:
            v = np.zeros((len(x), len(self.bin_centers)))
            for j in range(len(x)):
                for i in range(len(self.bin_centers)):
                    x1 = self.bin_centers[i] - self.bin_step / 2
                    x2 = self.bin_centers[i] + self.bin_step / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=self.sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

class DifferentiableRound(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale[0]

    def forward(self, x):
        # print(type(x))
        # print(type(self.scale))
        # print(type(torch.pi))
        # print(type(x - torch.sin(2 * torch.pi * x * self.scale) / (2 * torch.pi * self.scale)))
        return (x - torch.sin(2 * torch.pi * x * self.scale) / (2 * torch.pi * self.scale)).float()


def my_KLDivLoss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y += 1e-16
    n = y.shape[0]
    loss = loss_func(x, y) / n
    #print(loss)
    return loss

class TransparentProcessor(object):
    def __init__(self):
        super().__init__()
    
    def transform(self, x):
        return x.cpu().detach().numpy()
    
    def fit(self, x):
        return None
    
    def inverse_transform(self, x):
        return x

all_label_data = []
if "clinical" in modalities:
    clinical_names = np.load(os.path.join(args.datadir, "clinical_names.npy"), allow_pickle=True)
    # print(clinical_names)

for data in loader:
    data, metadata, _ = data
    # surface_lh = data["surface-lh"][0].detach().numpy()
    # surface_rh = data["surface-rh"][0].detach().numpy()
    if args.to_predict in metadata.keys():
        all_label_data.append(metadata[args.to_predict])
    else:
        index_to_predict = clinical_names.tolist().index(args.to_predict)
        all_label_data.append(data["clinical"][:, index_to_predict])
all_label_data = np.concatenate(all_label_data)
label_values = np.unique(all_label_data)
# plt.hist(all_label_data)

# all_label_data_valid = []
# for data in valid_loader:
#     all_label_data_valid.append(metadata[args.to_predict])
# all_label_data_valid = np.concatenate(all_label_data_valid)
# label_values_valid = np.unique(all_label_data_valid)
# plt.hist(all_label_data_valid)

# from nilearn import plotting
# from surfify.utils import icosahedron

# ico = icosahedron(order=args.ico_order)
# # for i in range(surface_lh.shape[-1]):
# i = 0
# plotting.plot_surf(ico, surface_lh[:, i])
# plotting.plot_surf(ico, surface_rh[:, i])

# print(other_other_mesh.coordinates.shape)
# print(other_other_mesh.faces.shape)

# plotting.plot_surf_roi(fsaverage["sphere_left"], roi_map=destrieux_atlas["map_left"], hemi="left",
#                        view="lateral", bg_map=fsaverage["sulc_left"], bg_on_data=True, darkness=0.2)

# plotting.plot_surf_roi(fsaverage["pial_left"], roi_map=destrieux_atlas["map_left"], hemi="left",
#                        view="lateral", bg_map=fsaverage["sulc_left"], bg_on_data=True, darkness=0.2)
# plotting.show()

# plt.show()

def corr_metric(y_true, y_pred):
    mat = np.concatenate([y_true[:, np.newaxis], y_pred[:, np.newaxis]], axis=1)
    corr_mat = np.corrcoef(mat, rowvar=False)
    return corr_mat[0, 1]

output_activation = nn.Identity()
hidden_dim = 128
tensor_type = "float"
out_to_pred_func = lambda x: x
out_to_real_pred_func = lambda x: x
root_mean_squared_error = lambda x, y: mean_squared_error(x, y, squared=False)
evaluation_against_real_metric = {"real_rmse": root_mean_squared_error, "real_mae": mean_absolute_error, "r2": r2_score, "correlation": corr_metric}
if args.method == "regression":
    output_dim = 1
    
    label_prepro = StandardScaler()
    label_prepro.fit(all_label_data[:, np.newaxis])
    # output_activation = DifferentiableRound(label_prepro.scale_)
    
    evaluation_metrics = {"rmse": root_mean_squared_error, "mae": mean_absolute_error}
    regressor = Ridge()
    out_to_real_pred_func = lambda x: label_prepro.inverse_transform(x).squeeze()
else:
    output_dim = len(label_values)
    evaluation_metrics = {"accuracy": accuracy_score}
    tensor_type = "long"
    n_bins = 3
    label_prepro = TransparentProcessor()
    out_to_real_pred_func = lambda x : x.argmax(1).cpu().detach().numpy()
    if any([type(value) is np.str_ for value in label_values]):
        label_prepro = OrdinalEncoder()
        label_prepro.fit(all_label_data[:, np.newaxis])
        out_to_real_pred_func = lambda x : label_prepro.inverse_transform(
            x.argmax(1).cpu().detach().unsqueeze(1).numpy()).squeeze()
    if output_dim > n_bins:
        label_prepro = KBinsDiscretizer(n_bins=n_bins, encode="ordinal")
        label_prepro.fit(all_label_data[:, np.newaxis])
        print(label_prepro.bin_edges_)
        output_dim = n_bins
    evaluation_against_real_metric = {}
    out_to_pred_func = lambda x: x.argmax(1).cpu().detach().numpy()
    regressor = LogisticRegression()

im_shape = next(iter(loader))[0]["surface-lh"][0].shape
n_features = im_shape[0]





all_metrics = {}
for name in evaluation_metrics.keys():
    all_metrics[name] = []
for name in evaluation_against_real_metric.keys():
    all_metrics[name] = []

run_name = ("deepint_evaluate_representations_to_predict_{}_{}_predict_via_{}"
            "_pretrained_{}").format(
                args.data, args.to_predict, args.method, args.pretrained_setup)

resdir = os.path.join(resdir, run_name)
if not os.path.isdir(resdir):
    os.makedirs(resdir)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

args.projector = "256-512-512"
args.lambd = 0.0051
class BarlowTwins(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()
        self.args = args
        self.backbone = backbone

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

conv_filters = [int(num) for num in args.conv_filters.split("-")]
backbone = SphericalHemiFusionEncoder(
            n_features, args.ico_order, args.latent_dim, fusion_level=args.fusion_level,
            conv_flts=conv_filters, activation=args.activation,
            batch_norm=args.batch_norm, conv_mode=args.conv,
            cachedir=os.path.join(args.outdir, "cached_ico_infos"))
on_the_fly_transform = dict()
for modality in modalities:
    transformer = Transformer(["hard", "soft"])
    if args.standardize:
        transformer.register(scalers[modality])
    if use_grid:
        channels_to_switch = (2, 0, 1)
        transformer.register(Permute(channels_to_switch))
    if args.normalize:
        transformer.register(Normalize())
    if args.gaussian_blur_augment:
        if use_grid:
            # transformer.register(RescaleAsImage(metrics))
            transformer.register(ToPILImage)
            transformer.register(GaussianBlur(), pipeline="hard")
            transformer.register(GaussianBlur(), probability=0.1, pipeline="soft")
            transformer.register(transforms.ToTensor())
        else:
            ico = backbone.ico[args.ico_order]
            transform = SphericalBlur(
                ico.vertices, ico.triangles, None,
                sigma=(0.1, 1))
            transformer.register(transform, pipeline="hard")
            transformer.register(transform, probability=0.1, pipeline="soft")
    if args.cutout:
        transform = Cutout(patch_size=np.ceil(np.array(input_shape)/4))
        if not use_grid:
            ico = backbone.ico[args.ico_order]
            # We want to set the maximum size size to barely 1/4 of the 
            # vertices. Since at each order, the number of vertices is
            # multiplied by 4, the neighborhood to be considered is of
            # order input_order - 1
            transform = SphericalRandomCut(
                ico.vertices, ico.triangles, ico.neighbor_indices,
                patch_size=args.ico_order - 1)
        transformer.register(transform, pipeline="hard")
        transformer.register(transform, probability=0.5, pipeline="soft")
    on_the_fly_transform[modality] = transformer

other_dataset = DataManager(dataset=args.data, datasetdir=args.datadir,
    modalities=modalities, validation=n_folds,
    stratify_on=["sex", "age"], discretize=["age"],
    transform=transform, on_the_fly_transform=on_the_fly_transform,
    overwrite=False, **kwargs)

class SelectNthDim(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x[self.dim]

for fold in range(n_folds):

    train_loader = torch.utils.data.DataLoader(
        dataset["train"][fold]["train"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset["train"][fold]["valid"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
        shuffle=True)
    if args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            dataset["train"]["all"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
            shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            dataset["test"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
            shuffle=True)

        other_train_loader = torch.utils.data.DataLoader(
            other_dataset["train"]["all"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
            shuffle=True)
        other_valid_loader = torch.utils.data.DataLoader(
            other_dataset["test"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
            shuffle=True)

    if use_grid:
        encoder = HemiFusionEncoder(n_features, input_size, args.latent_dim,
                                    fusion_level=args.fusion_level,
                                    conv_flts=conv_filters,
                                    activation=args.activation,
                                    batch_norm=args.batch_norm,
                                    return_dist=False)
    else:
        encoder = SphericalHemiFusionEncoder(
            n_features, args.ico_order, args.latent_dim, fusion_level=args.fusion_level,
            conv_flts=conv_filters, activation=args.activation,
            batch_norm=args.batch_norm, conv_mode=args.conv,
            cachedir=os.path.join(args.outdir, "cached_ico_infos"))
        other_encoder = SphericalHemiFusionEncoder(
            n_features, args.ico_order, args.latent_dim, fusion_level=args.fusion_level,
            conv_flts=conv_filters, activation=args.activation,
            batch_norm=args.batch_norm, conv_mode=args.conv,
            cachedir=os.path.join(args.outdir, "cached_ico_infos"))

    if checkpoint is not None:
        print("loading encoder")
        encoder.load_state_dict(checkpoint)
    
    
    if use_grid:
        encoder = nn.Sequential(encoder, SelectNthDim(0))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = encoder.to(device)
    other_model = BarlowTwins(args, encoder).to(device)
    other_other_model = BarlowTwins(args, other_encoder).to(device)
    other_cp = torch.load(pretrained_path.replace("encoder.pth", "barlow.pth"))
    other_other_model.load_state_dict(other_cp)
    other_model.projector = other_other_model.projector
    other_model.bn = other_other_model.bn

    # encoder = SphericalHemiFusionEncoder(
    #         n_features, args.ico_order, args.latent_dim, fusion_level=args.fusion_level,
    #         conv_flts=conv_filters, activation=args.activation,
    #         batch_norm=args.batch_norm, conv_mode=args.conv,
    #         cachedir=os.path.join(args.outdir, "cached_ico_infos")).to(device)
    # other_model.backbone = encoder
    # print(model)
    # print("Number of trainable parameters : ",
    #     sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Validation
    model.eval()
    other_model.eval()
    latents = []
    transformed_ys = []
    ys = []
    full_loss = 0
    for step, x in enumerate(train_loader):
        x, metadata, _ = x
        left_x = x["surface-lh"].float().to(device, non_blocking=True)
        right_x = x["surface-rh"].float().to(device, non_blocking=True)
        if args.to_predict in metadata.keys():
            y = metadata[args.to_predict]
        else:
            y = x["clinical"][:, index_to_predict]
        new_y = label_prepro.transform(np.array(y)[:, np.newaxis])
        transformed_ys.append(new_y)
        ys.append(y)
        with torch.cuda.amp.autocast():
            X = (left_x, right_x)
            latents.append(other_model.backbone(X).squeeze().detach().cpu().numpy())
            loss = other_model(X, X)
        full_loss += loss.item()
    y = np.concatenate(transformed_ys)
    real_y = np.concatenate(ys)
    X = np.concatenate(latents)
    regressor.fit(X, y)

    y_hat = regressor.predict(X)
    real_preds = out_to_real_pred_func(y_hat)
    for name, metric in evaluation_against_real_metric.items():
        print(name, metric(real_y, real_preds))

    valid_latents = []
    valid_ys = []
    valid_transformed_ys = []
    full_valid_loss = 0
    for step, x in enumerate(valid_loader):
        x, metadata, _ = x
        left_x = x["surface-lh"].float().to(device, non_blocking=True)
        right_x = x["surface-rh"].float().to(device, non_blocking=True)
        if args.to_predict in metadata.keys():
            y = metadata[args.to_predict]
        else:
            y = x["clinical"][:, index_to_predict]
        new_y = label_prepro.transform(np.array(y)[:, np.newaxis])
        valid_ys.append(y)
        valid_transformed_ys.append(new_y)
        with torch.cuda.amp.autocast():
            X = (left_x, right_x)
            if use_mlp:
                X = torch.cat(X, dim=1).view((len(left_x), -1))
            valid_latents.append(other_model.backbone(X).squeeze().detach().cpu().numpy())
            valid_loss = other_model(X, X)
        full_valid_loss += valid_loss.item()
    
    X_valid = np.concatenate(valid_latents)
    y_valid = np.concatenate(valid_transformed_ys)
    real_y_valid = np.concatenate(valid_ys)

    print(full_loss / len(dataset["train"]["all"]))
    print(full_valid_loss / len(dataset["test"]))
    y_hat = regressor.predict(X_valid)
   
    full_loss = 0
    for step, x in enumerate(other_train_loader):
        x, metadata, _ = x
        left_x_1, left_x_2 = x["surface-lh"]
        right_x_1, right_x_2 = x["surface-rh"]
        left_x_1 = left_x_1.float().to(device)
        left_x_2 = left_x_2.float().to(device)
        right_x_1 = right_x_1.float().to(device)
        right_x_2 = right_x_2.float().to(device)
        with torch.cuda.amp.autocast():
            loss = other_model((left_x_1, right_x_1), (left_x_2, right_x_2))
        full_loss += loss.item()
   
    full_valid_loss = 0
    for step, x in enumerate(other_valid_loader):
        x, metadata, _ = x
        left_x_1, left_x_2 = x["surface-lh"]
        right_x_1, right_x_2 = x["surface-rh"]
        left_x_1 = left_x_1.float().to(device)
        left_x_2 = left_x_2.float().to(device)
        right_x_1 = right_x_1.float().to(device)
        right_x_2 = right_x_2.float().to(device)
        with torch.cuda.amp.autocast():
            valid_loss = other_model((left_x_1, right_x_1), (left_x_2, right_x_2))
        full_valid_loss += valid_loss.item()

    print(full_loss / len(dataset["train"]["all"]))
    print(full_valid_loss / len(dataset["test"]))

    preds = out_to_pred_func(y_hat)
    real_preds = out_to_real_pred_func(y_hat)

    for name, metric in evaluation_metrics.items():
        all_metrics[name].append(metric(y_valid, preds))
    for name, metric in evaluation_against_real_metric.items():
        all_metrics[name].append(metric(real_y_valid, real_preds))
    break
average_metrics = {}
std_metrics = {}
for metric in all_metrics.keys():
    average_metrics[metric] = np.mean(all_metrics[metric])
    std_metrics[metric] = np.std(all_metrics[metric])

if not args.evaluate:
    groups = {"closeness": ["real_mae", "real_rmse"], "coherence": ["r2", "correlation"]}
    limit_per_group = {"closeness": (0, 10), "coherence": (0, 1)}
    for name, group in groups.items():
        plt.figure()
        for metric in group:
            values = average_metrics[metric]
            plt.plot(range(args.epochs), values, label=metric)
        plt.title("Average validation metrics")
        plt.ylim(limit_per_group[name])
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(os.path.join(
            resdir,"validation_metrics_{}.png".format(name)))

    best_epochs_per_metric = {}
    best_values_per_metric = {}
    best_stds_per_metric = {}
    the_lower_the_better = ["real_mae", "real_rmse"]
    for metric in ["real_mae", "real_rmse", "r2", "correlation"]:
        sorted_epochs = np.argsort(average_metrics[metric])
        sorted_values = np.sort(average_metrics[metric])
        if metric in the_lower_the_better:
            best_epochs = sorted_epochs[:5]
            best_values = sorted_values[:5]
        else:
            best_epochs = sorted_epochs[::-1][:5]
            best_values = sorted_values[::-1][:5]
        best_epochs_per_metric[metric] = best_epochs.tolist()
        best_values_per_metric[metric] = best_values.tolist()
        best_stds_per_metric[metric] = std_metrics[metric][best_epochs].tolist()

    with open(os.path.join(resdir, 'best_values.json'), 'w') as fp:
        json.dump(best_values_per_metric, fp)

    with open(os.path.join(resdir, 'best_epochs.json'), 'w') as fp:
        json.dump(best_epochs_per_metric, fp)

    with open(os.path.join(resdir, 'best_stds.json'), 'w') as fp:
        json.dump(best_stds_per_metric, fp)
else:
    final_value_per_metric = {}
    final_std_per_metric = {}
    for metric in ["real_mae", "real_rmse", "r2", "correlation"]:
        final_value_per_metric[metric] = average_metrics[metric]
        final_std_per_metric[metric] = std_metrics[metric]

    with open(os.path.join(resdir, 'final_values.json'), 'w') as fp:
        json.dump(final_value_per_metric, fp)
    
    with open(os.path.join(resdir, 'final_stds.json'), 'w') as fp:
        json.dump(final_std_per_metric, fp)