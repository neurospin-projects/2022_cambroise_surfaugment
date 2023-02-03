import argparse
import json
import os
import sys
import time
import parse
import numpy as np
from tqdm import tqdm
import joblib
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
from brainboard import Board

from multimodaldatasets.datasets import DataManager, DataLoaderWithBatchAugmentation
from augmentations import Permute, Normalize, GaussianBlur, RescaleAsImage, PermuteBeetweenModalities, Bootstrapping, Reshape, Transformer


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
    "--epochs", default=100, type=int, metavar="N",
    help="number of total epochs to run.")
parser.add_argument(
    "--batch-size", "-bs", default=128, type=int, metavar="N",
    help="mini-batch size.")
parser.add_argument(
    "--learning-rate", "-lr", default=1e-4, type=float, metavar="LR",
    help="base learning rate.")
parser.add_argument(
    "--weight-decay", "-wd", default=1e-6, type=float, metavar="W",
    help="weight decay.")
parser.add_argument(
    "--conv-filters", default="64-128-128-256-256", type=str, metavar="F",
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
    "--augment-data", action="store_true",
    help="optionnally uses augmentations during training."
)
parser.add_argument(
    "--to-predict", default="age",
    help="the name of the variable to predict.")
parser.add_argument(
    "--method", default="regression", choices=("regression", "classification", "distribution"),
    help="the prediction method.")
parser.add_argument(
    "--loss", default="mse", choices=("mse", "l1"),
    help="the loss function.")
parser.add_argument(
    "--bin-step", default=1, type=int,
    help="the size of a bin when method is 'distribution'.")
parser.add_argument(
    "--sigma", default=1, type=float,
    help="the soft labels' gaussian's sigma when method is 'distribution'.")
parser.add_argument(
    "--dropout-rate", "-dr", default=0, type=float,
    help="the dropout rate applied before predictor.")
parser.add_argument(
    "--pretrained-setup", default="None",
    help="the path to the pretrained encoder.")
parser.add_argument(
    "--setups-file", default="None",
    help="the path to the file linking the setups to the pretrained encoder's path.")
parser.add_argument(
    "--freeze-up-to", default=0, type=int,
    help="optionnally freezes the backbone network's weights up to some layer."
)
parser.add_argument(
    "--momentum", default=0, type=float,
    help="Momentum of the SGD optimizer. If 0 uses Adam optimizer."
)
parser.add_argument(
    "--evaluate", action="store_true",
    help="Uses the whole training set to train once, and evaluates on the hold-out test set."
)
parser.add_argument(
    "--batch-augment", "-ba", default=0.0, type=float,
    help="optionnally uses batch augmentation.")
parser.add_argument(
    "--inter-modal-augment", "-ima", default=0.0, type=float,
    help="optionnally uses inter modality augment.")
parser.add_argument(
    "--gaussian-blur-augment", "-gba", default=0.0, type=float,
    help="optionnally uses inter modality augment.")
args = parser.parse_args()
args.ngpus_per_node = torch.cuda.device_count()
args.conv_filters = [int(item) for item in args.conv_filters.split("-")]

# Prepare process
setup_logging(level="info", logfile=None)
checkpoint_dir = os.path.join(args.outdir, "predict_{}".format(args.to_predict), "checkpoints")
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
grid_size = 192

limits_per_metric = {"curv": [-5, 5]}
def clip_values(textures, channel_dim=-1):
    metric_idx_to_clip = {key: metrics.index(key) for key in limits_per_metric}
    for key, idx in metric_idx_to_clip.items():
        clipped_values = np.expand_dims(np.clip(np.take(textures, idx, axis=channel_dim), *limits_per_metric[key]), axis=channel_dim)
        where = np.ones([1] * len(textures.shape), dtype=np.int64) * idx
        np.put_along_axis(textures, where, clipped_values, axis=channel_dim)
    return textures


def transform_to_grid_wrapper(order=args.ico_order, grid_size=grid_size, channel_dim=1,
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
    input_shape = 2 * len(metrics) * len(icosahedron(7)[0])

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
    args.normalize = False
    args.batch_norm = False
    args.conv_filters = "64-128-128-256-256"
    args.latent_dim = 64
    args.gaussian_blur_augment = False
    args.cutout = False

input_shape = ((grid_size, grid_size, len(metrics)) if use_grid else
               (len(metrics), len(ico_verts)))

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

# normalize = True
# if args.inter_modal_augment > 0 or args.batch_augment > 0:
#     normalize = False


# if args.inter_modal_augment > 0:
#     normalizer = Normalize() if args.batch_augment == 0 else None
#     on_the_fly_inter_transform = PermuteBeetweenModalities(
#         args.inter_modal_augment, 0.3, ("surface-lh", "surface-rh"),
#         normalizer)

batch_transforms = None
batch_transforms_valid = None

# downsampler = wrapper_data_downsampler(args.outdir, to_order=args.ico_order)

kwargs = {
    "surface-rh": {"metrics": metrics,
                   "z_score": False},
    "surface-lh": {"metrics": metrics,
                   "z_score": False},
    "test_size": 0.2
}

if args.data in ["hbn", "euaims"]:
    kwargs["surface-lh"]["symetrized"] = True

    kwargs["surface-rh"]["symetrized"] = True

    modalities.append("clinical")

if args.data == "openbhb":
    kwargs["test_size"] = None

n_folds = 5

dataset = DataManager(dataset=args.data, datasetdir=args.datadir,
                      modalities=modalities, validation=n_folds,
                      stratify_on=["sex", "age"], discretize=["age"],
                      transform=transform, on_the_fly_transform=on_the_fly_transform,
                      overwrite=False, **kwargs)



loader = torch.utils.data.DataLoader(
    dataset["train"]["all"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset["test"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
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
out_to_pred_func = lambda x: x.cpu().detach().numpy()
out_to_real_pred_func = lambda x: x.cpu().detach().numpy()
root_mean_squared_error = lambda x, y: mean_squared_error(x, y, squared=False)
evaluation_against_real_metric = {"real_rmse": root_mean_squared_error, "real_mae": mean_absolute_error, "r2": r2_score, "correlation": corr_metric}
if args.method == "regression":
    output_dim = 1
    
    label_prepro = StandardScaler()
    label_prepro.fit(all_label_data[:, np.newaxis])
    # output_activation = DifferentiableRound(label_prepro.scale_)
    
    criterion = nn.MSELoss()
    if args.loss == "l1":
        criterion = nn.L1Loss()
    evaluation_metrics = {"rmse": root_mean_squared_error, "mae": mean_absolute_error}
    out_to_real_pred_func = lambda x: label_prepro.inverse_transform(x.cpu().detach().unsqueeze(1)).squeeze()
elif args.method in ["classification", "distribution"]:

    output_dim = len(label_values)
    evaluation_metrics = {"accuracy": accuracy_score}
    if args.method == "classification":
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
        criterion = nn.CrossEntropyLoss()
        evaluation_against_real_metric = {}
        out_to_pred_func = lambda x: x.argmax(1).cpu().detach().numpy()
    else:
        bin_step = args.bin_step
        sigma = args.sigma
        output_activation = nn.LogSoftmax(dim=1)
        label_prepro = SoftBiner(bin_step, sigma)
        label_prepro.fit(all_label_data)
        output_dim = len(label_prepro.bin_centers)
        criterion = nn.KLDivLoss(reduction="batchmean")
        criterion = my_KLDivLoss
        out_to_real_pred_func = lambda x:  x.exp().cpu().detach().numpy() @ label_prepro.bin_centers
        evaluation_metrics = {}

im_shape = next(iter(loader))[0]["surface-lh"][0].shape
n_features = im_shape[0]

linear_modeling = False # args.evaluate
# Linear modeling
if linear_modeling:

    alphas = [0.1]
    perfs = {metric: [[] for _ in range(len(alphas))] for metric in evaluation_against_real_metric}
    stds = {}

    for fold in range(n_folds):
        train_loader = torch.utils.data.DataLoader(
            dataset["train"][fold]["train"], batch_size=args.batch_size, num_workers=4, pin_memory=True,
            shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            dataset["train"][fold]["valid"], batch_size=args.batch_size, num_workers=4, pin_memory=True,
            shuffle=True)
        if args.evaluate:
            train_loader = loader
            valid_loader = test_loader
        X = []
        Y = []
        metadata = []
        for data in train_loader:
            data, metadata, _ = data
            n_subj = len(data["surface-lh"])
            lh = data["surface-lh"].view(n_subj, -1).detach().numpy()
            rh = data["surface-rh"].view(n_subj, -1).detach().numpy()
            if args.to_predict in metadata.keys():
                y = metadata[args.to_predict].detach().numpy()
            else:
                y = data["clinical"][:, index_to_predict].detach().numpy()
            meta = pd.DataFrame(metadata)
            X.append(np.concatenate([lh, rh], axis=1))
            Y.append(y)
            metadata.append(meta)
        X = np.concatenate(X, axis=0)
        Y = np.concatenate(Y, axis=0)
        metadata = pd.concat(metadata, axis=0)

        # print(X.shape)
        # print(Y.shape)
        nan_idx = np.isnan(Y)
        Y = Y[~nan_idx]
        X = X[~nan_idx]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        clinical_scaler = StandardScaler()
        # binarizer = KBinsDiscretizer(2, encode="ordinal", strategy="uniform")
        # Y = binarizer.fit_transform(Y)
        Y = clinical_scaler.fit_transform(Y[:, np.newaxis]).squeeze()

        X_valid = []
        Y_valid = []
        metadata_valid = []
        for data in valid_loader:
            data, metadata, _ = data
            n_subj = len(data["surface-lh"])
            lh = data["surface-lh"].view(n_subj, -1).detach().numpy()
            rh = data["surface-rh"].view(n_subj, -1).detach().numpy()
            if args.to_predict in metadata.keys():
                y = metadata[args.to_predict].detach().numpy()
            else:
                y = data["clinical"][:, index_to_predict].detach().numpy()
            meta = pd.DataFrame(metadata) 
            X_valid.append(np.concatenate([lh, rh], axis=1))
            Y_valid.append(y)
            metadata_valid.append(meta)
        X_valid = np.concatenate(X_valid, axis=0)
        Y_valid = np.concatenate(Y_valid, axis=0)
        metadata_valid = pd.concat(metadata_valid, axis=0)

        # print(np.isnan(Y_valid).sum())

        X_valid = scaler.transform(X_valid)
        Y_valid = clinical_scaler.transform(Y_valid[:, np.newaxis]).squeeze()
        print("Scaled and loaded")
        # Y_valid = binarizer.transform(Y_valid)

        # print("Predict clinical from surf")
        # for alpha in [0.1, 1, 10, 100, 1000]:#, 10000, 50000, 100000]:
        #     print("Regularisation : {}".format(alpha))
        #     for i in range(Y.shape[1]):
        #         print("predicting feature {}".format(i))
        #         predictor = Ridge(alpha)
        #         predictor.fit(X, Y[:, i])
        #         train_score = predictor.score(X, Y[:, i])
        #         test_score = predictor.score(X_valid, Y_valid[:, i])
        #         print("train mse : {}".format(mean_squared_error(predictor.predict(X), Y[:, i])))
        #         print("valid mse : {}".format(mean_squared_error(predictor.predict(X_valid), Y_valid[:, i])))
        #         print("train mae : {}".format(mean_absolute_error(predictor.predict(X), Y[:, i])))
        #         print("valid mae : {}".format(mean_absolute_error(predictor.predict(X_valid), Y_valid[:, i])))
        #         print("train r2 : {}".format(train_score))
        #         print("valid r2 : {}".format(test_score))
        #         print("predictions : {}".format(predictor.predict(X_valid)))
        # print(Y)

        # print("Predict clinical from surf")
        for idx, alpha in enumerate(alphas):#, 10000, 50000, 100000]:
            # print("Regularisation : {}".format(alpha))
            # print("predicting {}".format(args.to_predict))
            predictor = Ridge(alpha)
            predictor.fit(X, Y)
            train_score = predictor.score(X, Y)
            test_score = predictor.score(X_valid, Y_valid)
            true_prediction = clinical_scaler.inverse_transform(predictor.predict(X)[:, np.newaxis]).squeeze()
            true_label = clinical_scaler.inverse_transform(Y[:, np.newaxis]).squeeze()
            true_prediction_valid = clinical_scaler.inverse_transform(predictor.predict(X_valid)[:, np.newaxis]).squeeze()
            true_label_valid = clinical_scaler.inverse_transform(Y_valid[:, np.newaxis]).squeeze()
            # print("train mse : {}".format(mean_squared_error(true_prediction, true_label)))
            # print("valid mse : {}".format(mean_squared_error(true_prediction_valid, true_label_valid)))
            # print("train mae : {}".format(mean_absolute_error(true_prediction, true_label)))
            # print("valid mae : {}".format(mean_absolute_error(true_prediction_valid, true_label_valid)))
            # print("train r2 : {}".format(train_score))
            # print("valid r2 : {}".format(test_score))
            for name, metric in evaluation_against_real_metric.items():
                perfs[name][idx].append(metric(true_label_valid, true_prediction_valid))
    
    for metric in evaluation_against_real_metric:
        stds[metric] = np.std(np.array(perfs[metric]), axis=1).tolist()
        perfs[metric] = np.mean(np.array(perfs[metric]), axis=1).tolist()
        
    print(stds)
    print(perfs)
    # print("Predict clinical from surf")
    # for alpha in [0.1, 1, 10, 100, 1000]:#, 10000, 50000, 100000]:
    #     print("Regularisation : {}".format(alpha))
    #     print("predicting {}".format(args.to_predict))
    #     predictor = Ridge(alpha)
    #     predictor.fit(X, Y)
    #     train_score = predictor.score(X, Y)
    #     test_score = predictor.score(X_valid, Y_valid)
    #     true_prediction = clinical_scaler.inverse_transform(predictor.predict(X)[:, np.newaxis]).squeeze()
    #     true_label = clinical_scaler.inverse_transform(Y[:, np.newaxis]).squeeze()
    #     true_prediction_valid = clinical_scaler.inverse_transform(predictor.predict(X_valid)[:, np.newaxis]).squeeze()
    #     true_label_valid = clinical_scaler.inverse_transform(Y_valid[:, np.newaxis]).squeeze()
    #     print("train mse : {}".format(mean_squared_error(true_prediction, true_label)))
    #     print("valid mse : {}".format(mean_squared_error(true_prediction_valid, true_label_valid)))
    #     print("train mae : {}".format(mean_absolute_error(true_prediction, true_label)))
    #     print("valid mae : {}".format(mean_absolute_error(true_prediction_valid, true_label_valid)))
    #     print("train r2 : {}".format(train_score))
    #     print("valid r2 : {}".format(test_score))
        # print("predictions : {}".format(true_prediction_valid))
    # print(true_label_valid)

    # print("Predict diagnostic from surf")
    # for alpha in [0.1, 1, 10, 100, 1000]:
    #     print("Regularisation : {}".format(alpha))
    #     print("predicting feature {}".format(i))
    #     predictor = LogisticRegression(C=1/alpha)
    #     labels = metadata["asd"].values - 1
    #     valid_labels = metadata_valid["asd"].values - 1
    #     predictor.fit(X, labels)
    #     train_score = predictor.score(X, labels)
    #     test_score = predictor.score(X_valid, valid_labels)
    #     print("train acc : {}".format(accuracy_score(predictor.predict(X), labels)))
    #     print("valid acc : {}".format(accuracy_score(predictor.predict(X_valid), valid_labels)))
    #     print("train recall : {}".format(recall_score(predictor.predict(X), labels)))
    #     print("valid recall : {}".format(recall_score(predictor.predict(X_valid), valid_labels)))
    #     print("train acc : {}".format(train_score))
    #     print("valid acc : {}".format(test_score))
    #     print("valid diagnosis : {}".format(valid_labels))
    #     print("predictions : {}".format(predictor.predict(X_valid)))


    # print("Predict diagnostic from clinical")
    # for alpha in [0.01, 0.1, 1, 10]:
    #     print("Regularisation : {}".format(alpha))
    #     print("predicting feature {}".format(i))
    #     predictor = LogisticRegression(C=1/alpha)
    #     labels = metadata["asd"].values - 1
    #     valid_labels = metadata_valid["asd"].values - 1
    #     predictor.fit(Y, labels)
    #     train_score = predictor.score(Y, labels)
    #     test_score = predictor.score(Y_valid, valid_labels)
    #     print("train acc : {}".format(accuracy_score(predictor.predict(Y), labels)))
    #     print("valid acc : {}".format(accuracy_score(predictor.predict(Y_valid), valid_labels)))
    #     print("train recall : {}".format(recall_score(predictor.predict(Y), labels)))
    #     print("valid recall : {}".format(recall_score(predictor.predict(Y_valid), valid_labels)))
    #     print("train acc : {}".format(train_score))
    #     print("valid acc : {}".format(test_score))
    #     print("valid diagnosis : {}".format(valid_labels))
    #     print("predictions : {}".format(predictor.predict(Y_valid)))

    # for idx in range(Y.shape[1]):
    #     print(np.corrcoef(np.concatenate([Y[:, idx, np.newaxis], metadata[["asd"]].values], axis=1), rowvar=False)[0, 1])

# print(pretrained_path)

activation = "ReLU"
all_metrics = {}
for name in evaluation_metrics.keys():
    all_metrics[name] = [[] for _ in range(args.epochs)]
for name in evaluation_against_real_metric.keys():
    all_metrics[name] = [[] for _ in range(args.epochs)]

run_name = ("deepint_predict_{}_{}_loss_{}_{}_features_fusion_{}_act_{}_bn_{}_conv_{}"
        "_latent_{}_lr_{}_momentum_{}_wd_{}_dr_{}_{}_epochs_predict_via_{}"
        "_pretrained_{}_freezed_{}_ba_{}_ima_{}_gba_{}").format(
            args.data, args.to_predict, criterion.__class__.__name__, n_features, args.fusion_level, activation,
            args.batch_norm, "-".join([str(s) for s in args.conv_filters]), args.latent_dim, args.learning_rate,
            args.momentum, args.weight_decay, args.dropout_rate, args.epochs, args.method,
            args.pretrained_setup, args.freeze_up_to, args.batch_augment, args.inter_modal_augment,
            args.gaussian_blur_augment)

checkpoint_dir = os.path.join(checkpoint_dir, run_name)
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

use_board = False
show_pbar = True

class SelectNthDim(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x[self.dim]

class ConcatAlongDim(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


for fold in range(n_folds):
    print("Training on fold {} / {}".format(fold + 1, n_folds))
    # if args.batch_augment > 0:
    #     original_dataset = DataManager(
    #         dataset=args.data, datasetdir=args.datadir,
    #         modalities=modalities, transform=transform,
    #         stratify_on=["sex", "age"], discretize=["age"],
    #         overwrite=False, on_the_fly_transform=None,
    #         on_the_fly_inter_transform=None,
    #         validation=n_folds, **kwargs)

    #     train_loader = torch.utils.data.DataLoader(
    #         original_dataset["train"][fold]["train"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #         shuffle=True)
    #     valid_loader = torch.utils.data.DataLoader(
    #         original_dataset["train"][fold]["valid"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #         shuffle=True)
    #     if args.evaluate:
    #         train_loader = torch.utils.data.DataLoader(
    #             original_dataset["train"]["all"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #             shuffle=True)
    #         valid_loader = torch.utils.data.DataLoader(
    #             original_dataset["test"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #             shuffle=True)
        
    #     if args.batch_augment > 0:
    #         groups = {}
    #         groups_valid = {}
    #         print("Initializing KNN...")
    #         for modality in ["surface-lh", "surface-rh"]:
    #             regressor = KNeighborsRegressor(n_neighbors=30)
    #             X, Y = [], []
    #             X_valid, Y_valid = [], []
    #             for x in train_loader:
    #                 x, metadata, _ = x
    #                 X += x[modality].view((len(x[modality]), -1)).tolist()
    #                 Y += metadata["age"].tolist()
        
    #             for x in valid_loader:
    #                 x, metadata, _ = x
    #                 X_valid += x[modality].view((len(x[modality]), -1)).tolist()
    #                 Y_valid += metadata["age"].tolist()

    #             # print("Scaling")
    #             scaler = StandardScaler()
    #             X = scaler.fit_transform(X)
    #             X_valid = scaler.transform(X_valid)
    #             # print("Scaled")

    #             # print("Reducting")
    #             reductor = PCA(30)
    #             X = reductor.fit_transform(X)
    #             X_valid = reductor.transform(X_valid)
    #             # print("Reducted")

    #             regressor.fit(X, Y)
    #             print(mean_absolute_error(Y, regressor.predict(X)))
    #             print(mean_absolute_error(Y_valid, regressor.predict(X_valid)))
    #             print(r2_score(Y, regressor.predict(X)))
    #             print(r2_score(Y_valid, regressor.predict(X_valid)))
    #             _, neigh_idx = regressor.kneighbors(X)
    #             _, neigh_idx_valid = regressor.kneighbors(X_valid)
    #             groups[modality] = neigh_idx
    #             groups_valid[modality] = neigh_idx_valid
    #         # print("Groups built.")

    #         batch_transforms = {
    #             "surface-lh": Bootstrapping(p=args.batch_augment, p_corrupt=0.3,
    #                                         groups=groups["surface-lh"]),
    #             "surface-rh": Bootstrapping(p=args.batch_augment, p_corrupt=0.3,
    #                                         groups=groups["surface-rh"]),
    #         }

    #         batch_transforms_valid = {
    #             "surface-lh": Bootstrapping(p=args.batch_augment, p_corrupt=0.3,
    #                                         groups=groups_valid["surface-lh"]),
    #             "surface-rh": Bootstrapping(p=args.batch_augment, p_corrupt=0.3,
    #                                         groups=groups_valid["surface-rh"]),
    #         }

    train_loader = DataLoaderWithBatchAugmentation(batch_transforms,
        dataset["train"][fold]["train"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
        shuffle=True)
    valid_loader = DataLoaderWithBatchAugmentation(batch_transforms_valid,
        dataset["train"][fold]["valid"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
        shuffle=True)


    # train_loader = torch.utils.data.DataLoader(
    #     dataset["train"][fold]["train"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #     shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(
    #     dataset["train"][fold]["valid"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #     shuffle=True)
    if args.evaluate:
        train_loader = DataLoaderWithBatchAugmentation(batch_transforms,
            dataset["train"]["all"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
            shuffle=True)
        valid_loader = DataLoaderWithBatchAugmentation(batch_transforms_valid,
            dataset["test"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
            shuffle=True)

    if use_mlp:
        modules = [nn.Linear(grid_size, args.latent_dim)]
        if args.batch_norm:
            modules.append(nn.BatchNorm1d(args.latent_dim))
        modules.append(nn.ReLU())
        encoder = nn.Sequential(*modules)
    elif use_grid:
        encoder = HemiFusionEncoder(n_features, grid_size, args.latent_dim,
                                fusion_level=args.fusion_level,
                                conv_flts=[int(num) for num in args.conv_filters.split("-")],
                                activation=activation,
                                batch_norm=args.batch_norm,
                                return_dist=False)
    else:
        encoder = SphericalHemiFusionEncoder(
            n_features, args.ico_order, args.latent_dim, fusion_level=args.fusion_level,
            conv_flts=args.conv_filters, activation=activation,
            batch_norm=args.batch_norm, conv_mode=args.conv,
            cachedir=os.path.join(args.outdir, "cached_ico_infos"))


    if checkpoint is not None:
        encoder.load_state_dict(checkpoint)

    all_encoder_params = list(encoder.parameters())
    assert np.abs(args.freeze_up_to) < len(all_encoder_params)
    if args.freeze_up_to != 0:
        number_of_layers_per_layer = 2 if not args.batch_norm else 3
        if args.freeze_up_to < 0:
            args.freeze_up_to = len(all_encoder_params) - args.freeze_up_to
        idx_to_freeze = []
        for idx in range(len(all_encoder_params)):
            fused = idx >= args.fusion_level * 2 * number_of_layers_per_layer
            idx_after_fusion = idx - args.fusion_level * 2 * number_of_layers_per_layer
            if not fused and idx % (args.fusion_level * number_of_layers_per_layer) < (args.freeze_up_to) * number_of_layers_per_layer:
                idx_to_freeze.append(idx)
            elif fused and idx_after_fusion < (args.freeze_up_to - args.fusion_level) * number_of_layers_per_layer:
                idx_to_freeze.append(idx)
        
        for idx, param in enumerate(all_encoder_params):
            if idx in idx_to_freeze:
                param.requires_grad = False

    predictor = nn.Sequential(*[
        # nn.Linear(args.latent_dim, hidden_dim),
        # nn.LeakyReLU(),
        # nn.Linear(hidden_dim, hidden_dim),
        # nn.LeakyReLU(),
        # nn.Linear(hidden_dim, hidden_dim),
        # nn.LeakyReLU(),
        nn.Dropout(args.dropout_rate),
        nn.Linear(args.latent_dim, output_dim),
        output_activation
    ])

    if use_grid:
        model = nn.Sequential(encoder, SelectNthDim(0), predictor)
    else:
        model = nn.Sequential(encoder, predictor)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # print(model)
    print("Number of trainable parameters : ",
        sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                        weight_decay=args.weight_decay)
    if args.momentum > 0:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay, momentum=args.momentum)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
    # scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.97 if epoch % 5 == 0 else 1)

    if args.epochs > 0 and use_board:
        board = Board(env=run_name + "_fold_{}".format(fold))

    start_epoch = 0
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        stats = dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"],
                    loss=0, validation_loss=0, validation_mse=0,
                    total_train_val_time=0, average_train_epoch_duration=0,
                    train_epoch_duration=0, average_valid_epoch_duration=0,
                    valid_epoch_duration=0)
        for name in evaluation_metrics.keys():
            stats[name] = 0
            stats["validation_" + name] = 0
        for name in evaluation_against_real_metric.keys():
            stats[name] = 0
            stats["validation_" + name] = 0

        # Training
        model.train()
        with tqdm(total=len(train_loader), desc="Training epoch : {} / {}".format(epoch, args.epochs),
                postfix={"loss": 0, "lr": args.learning_rate, "average time": 0}, disable=not show_pbar) as pbar:
            for step, x in enumerate(train_loader):
                # pbar.update(step + 1)
                x, metadata, _ = x
                start_batch_time = time.time()
                left_x = x["surface-lh"].float().to(device, non_blocking=True)
                right_x = x["surface-rh"].float().to(device, non_blocking=True)
                if args.to_predict in metadata.keys():
                    y = metadata[args.to_predict]
                else:
                    y = x["clinical"][:, index_to_predict]
                new_y = label_prepro.transform(np.array(y)[:, np.newaxis])
                new_y = getattr(torch.tensor(new_y), tensor_type)().squeeze()
                new_y = new_y.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    X = (left_x, right_x)
                    if use_mlp:
                        X = torch.cat(X, dim=1).view((len(left_x), -1))
                    y_hat = model(X).squeeze()
                    loss = criterion(y_hat, new_y)
                    preds = out_to_pred_func(y_hat)
                    real_preds = out_to_real_pred_func(y_hat)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                epoch_duration = time.time() - start_batch_time
                stats.update({
                    "loss": (stats["loss"] * step + loss.item()) / (step + 1),
                    "total_train_val_time": time.time() - start_time,
                    "average_train_epoch_duration": (stats["average_train_epoch_duration"] * step + 
                        epoch_duration) / (step + 1),
                    "train_epoch_duration": epoch_duration,
                })
                for name, metric in evaluation_metrics.items():
                    stats[name] = (stats[name] * step + metric(new_y.cpu().detach(), preds)) / (step + 1)
                for name, metric in evaluation_against_real_metric.items():
                    stats[name] = (stats[name] * step + metric(y.detach(), real_preds)) / (step + 1)
                pbar.set_postfix({
                    "loss": stats["loss"], "lr": stats["lr"],
                    "average_time": stats["average_train_epoch_duration"]})
                pbar.update(1)
            if use_board:
                board.update_plot("training loss", epoch, stats["loss"])
                for name in evaluation_metrics.keys():
                    board.update_plot(name, epoch, stats[name])
                for name in evaluation_against_real_metric.keys():
                    board.update_plot(name, epoch, stats[name])

        # Validation
        model.eval()
        with tqdm(total=len(valid_loader), desc="Validation epoch : {} / {}".format(epoch, args.epochs),
                postfix={"loss": 0, "average time": 0}, disable=not show_pbar) as pbar:
            for step, x in enumerate(valid_loader):
                x, metadata, _ = x
                start_batch_time = time.time()
                left_x = x["surface-lh"].float().to(device, non_blocking=True)
                right_x = x["surface-rh"].float().to(device, non_blocking=True)
                if args.to_predict in metadata.keys():
                    y = metadata[args.to_predict]
                else:
                    y = x["clinical"][:, index_to_predict]
                new_y = label_prepro.transform(np.array(y)[:, np.newaxis])
                new_y = getattr(torch.tensor(new_y), tensor_type)().squeeze()
                new_y = new_y.to(device, non_blocking=True)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        X = (left_x, right_x)
                        if use_mlp:
                            X = torch.cat(X, dim=1).view((len(left_x), -1))
                        y_hat = model(X).squeeze()
                        loss = criterion(y_hat, new_y)
                        preds = out_to_pred_func(y_hat)
                        real_preds = out_to_real_pred_func(y_hat)

                epoch_duration = time.time() - start_batch_time
                stats.update({
                    "validation_loss": (stats["validation_loss"] * step + loss.item()) / (step + 1),
                    "total_train_val_time": time.time() - start_time,
                    "average_valid_epoch_duration": (stats["average_valid_epoch_duration"] * step + 
                        epoch_duration) / (step + 1),
                    "valid_epoch_duration": epoch_duration,
                })
                for name, metric in evaluation_metrics.items():
                    stats["validation_" + name] = (stats["validation_" + name] * step + metric(new_y.cpu().detach(), preds)) / (step + 1)
                for name, metric in evaluation_against_real_metric.items():
                    stats["validation_" + name] = (stats["validation_" + name] * step + metric(y.detach(), real_preds)) / (step + 1)
                pbar.set_postfix({
                        "loss": stats["validation_loss"],
                        "average_time": stats["average_valid_epoch_duration"]})
                pbar.update(1)
            
        if use_board:
            board.update_plot("validation loss", epoch, stats["validation_loss"])
        for name in evaluation_metrics.keys():
            all_metrics[name][epoch].append(stats["validation_" + name])
            if use_board:
                board.update_plot("validation " + name, epoch, stats["validation_" + name])
        for name in evaluation_against_real_metric.keys():
            all_metrics[name][epoch].append(stats["validation_" + name])
            if use_board:
                board.update_plot("validation " + name, epoch, stats["validation_" + name])

        print(json.dumps(stats), file=stats_file)
        if args.evaluate and epoch == args.epochs - 1:
            print(stats)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))
        # scheduler.step()
    #     state = dict(epoch=epoch + 1,
    #                 model=model.state_dict(),
    #                 optimizer=optimizer.state_dict())#,
    #                 #  scheduler=scheduler.state_dict())
    #     torch.save(state, checkpoint_file)
    # torch.save(model.state_dict(), model_file)

average_metrics = {}
std_metrics = {}
for metric in all_metrics.keys():
    average_metrics[metric] = np.mean(all_metrics[metric], axis=1)
    std_metrics[metric] = np.std(all_metrics[metric], axis=1)

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
            checkpoint_dir,"validation_metrics_{}.png".format(name)))

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

    with open(os.path.join(checkpoint_dir, 'best_values.json'), 'w') as fp:
        json.dump(best_values_per_metric, fp)

    with open(os.path.join(checkpoint_dir, 'best_epochs.json'), 'w') as fp:
        json.dump(best_epochs_per_metric, fp)

    with open(os.path.join(checkpoint_dir, 'best_stds.json'), 'w') as fp:
        json.dump(best_stds_per_metric, fp)
else:
    final_value_per_metric = {}
    final_std_per_metric = {}
    for metric in ["real_mae", "real_rmse", "r2", "correlation"]:
        final_value_per_metric[metric] = average_metrics[metric][-1]
        final_std_per_metric[metric] = std_metrics[metric][-1]

    with open(os.path.join(checkpoint_dir, 'final_values.json'), 'w') as fp:
        json.dump(final_value_per_metric, fp)
    
    with open(os.path.join(checkpoint_dir, 'final_stds.json'), 'w') as fp:
        json.dump(final_std_per_metric, fp)

# checkpoint = torch.load(checkpoint_file)
# model.load_state_dict(checkpoint["model"])
    