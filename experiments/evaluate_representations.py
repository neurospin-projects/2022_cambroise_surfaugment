import argparse
import json
import os
import sys
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, OrdinalEncoder
from sklearn.linear_model import Ridge, LogisticRegression
import pandas as pd
import torch
from torch import nn, optim
from torchvision import transforms
from surfify.models import SphericalHemiFusionEncoder
from surfify.utils import setup_logging, icosahedron, downsample_data, downsample

from multimodaldatasets.datasets import DataManager
from augmentations import Normalize, Reshape, Transformer
import parse


# Get user parameters
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
    "--pretrained-setup", default="None",
    help="the pretrained encoder's id.")
parser.add_argument(
    "--setups-file", default="None",
    help="the path to the file linking the setups to the pretrained encoder's path.")
parser.add_argument(
    "--ico-order", default=6, type=int,
    help="the icosahedron order.")
args = parser.parse_args()
args.conv = "DiNe"

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
transform = None

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

checkpoint = None
if args.pretrained_setup != "None":
    assert args.setups_file != "None"
    setups = pd.read_table(args.setups_file)
    params, epoch = setups[setups["id"] == int(args.pretrained_setup)][["args", "best_epoch"]].values[0]
    pretrained_path = os.path.join(
            "/".join(args.setups_file.split("/")[:-1]),
            "checkpoints", args.pretrained_setup,
            "encoder.pth")
    if int(args.pretrained_setup) < 10000:
        pretrained_path = params
    args = params_from_args(params)
    max_epoch = int(params.split("_epochs")[0].split("_")[-1])
    if epoch != max_epoch:
        pretrained_path = pretrained_path.replace(
            "encoder.pth", "model_epoch_{}.pth".format(int(epoch)))
    checkpoint = torch.load(pretrained_path)
    if epoch != max_epoch:
        checkpoint = encoder_cp_from_model_cp(checkpoint)
else:
    args.n_features = len(metrics)
    args.fusion_level = 1
    args.activation = "ReLU"
    args.standardize = True
    args.normalize = False
    args.batch_norm = False
    args.conv_filters = "64-128-128-256-256"
    args.latent_dim = 64
    args.blur = False
    args.cutout = False
    args.noise = False
    args.ico_order = 6
args.batch_size = 32

print(args)

use_mlp = False
if use_mlp:
    input_size = 2 * len(metrics) * len(icosahedron(args.ico_order)[0])

input_shape = (len(metrics), len(icosahedron(args.ico_order)[0]))

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
    if args.normalize:
        transformer.register(Normalize())
    on_the_fly_transform[modality] = transformer


# downsampler = wrapper_data_downsampler(args.outdir, to_order=args.ico_order)

kwargs = {
    "surface-rh": {"metrics": metrics},
    "surface-lh": {"metrics": metrics},
}

if args.data in ["hbn", "euaims"]:
    kwargs["surface-lh"]["symetrized"] = True
    kwargs["surface-rh"]["symetrized"] = True

    modalities.append("clinical")



dataset = DataManager(
    dataset=args.data, datasetdir=args.datadir, modalities=modalities,
    stratify=["sex", "age", "site"], discretize=["age"],
    transform=transform, on_the_fly_transform=on_the_fly_transform,
    overwrite=False, test_size="defaults", **kwargs)
if args.data_train == args.data:
    dataset.create_val_from_test(
        val_size=0.5, stratify=["sex", "age", "site"], discretize=["age"])


loader = torch.utils.data.DataLoader(
    dataset["train"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    shuffle=True)

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
    if args.to_predict in metadata.keys():
        all_label_data.append(metadata[args.to_predict])
    else:
        index_to_predict = clinical_names.tolist().index(args.to_predict)
        all_label_data.append(data["clinical"][:, index_to_predict])
all_label_data = np.concatenate(all_label_data)
label_values = np.unique(all_label_data)
# plt.hist(all_label_data)

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

conv_filters = [int(num) for num in args.conv_filters.split("-")]


train_loader = torch.utils.data.DataLoader(
    dataset["train"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    shuffle=True)

test_dataset = dataset["test"]
if args.data_train == args.data:
    test_dataset = test_dataset["test"]
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, num_workers=6, pin_memory=True,
    shuffle=True)

encoder = SphericalHemiFusionEncoder(
    n_features, args.ico_order, args.latent_dim, fusion_level=args.fusion_level,
    conv_flts=conv_filters, activation=args.activation,
    batch_norm=args.batch_norm, conv_mode=args.conv,
    cachedir=os.path.join(args.outdir, "cached_ico_infos"))

if checkpoint is not None:
    print("loading encoder")
    encoder.load_state_dict(checkpoint)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = encoder.to(device)
   
    # print(model)
    # print("Number of trainable parameters : ",
    #     sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Validation
model.eval()
latents = []
transformed_ys = []
ys = []
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
        latents.append(model(X).squeeze().detach().cpu().numpy())
y = np.concatenate(transformed_ys)
real_y = np.concatenate(ys)
X = np.concatenate(latents)
regressor.fit(X, y)

y_hat = regressor.predict(X)
real_preds = out_to_real_pred_func(y_hat)
for name, metric in evaluation_against_real_metric.items():
    print(name, metric(real_y, real_preds))

test_latents = []
test_ys = []
test_transformed_ys = []
for step, x in enumerate(test_loader):
    x, metadata, _ = x
    left_x = x["surface-lh"].float().to(device, non_blocking=True)
    right_x = x["surface-rh"].float().to(device, non_blocking=True)
    if args.to_predict in metadata.keys():
        y = metadata[args.to_predict]
    else:
        y = x["clinical"][:, index_to_predict]
    new_y = label_prepro.transform(np.array(y)[:, np.newaxis])
    test_ys.append(y)
    test_transformed_ys.append(new_y)
    with torch.cuda.amp.autocast():
        X = (left_x, right_x)
        if use_mlp:
            X = torch.cat(X, dim=1).view((len(left_x), -1))
        test_latents.append(model(X).squeeze().detach().cpu().numpy())

X_test = np.concatenate(test_latents)
y_test = np.concatenate(test_transformed_ys)
real_y_test = np.concatenate(test_ys)

y_hat = regressor.predict(X_test)

preds = out_to_pred_func(y_hat)
real_preds = out_to_real_pred_func(y_hat)

for name, metric in evaluation_metrics.items():
    all_metrics[name].append(metric(y_test, preds))
for name, metric in evaluation_against_real_metric.items():
    all_metrics[name].append(metric(real_y_test, real_preds))

average_metrics = {}
std_metrics = {}
for metric in all_metrics.keys():
    average_metrics[metric] = np.mean(all_metrics[metric])
    std_metrics[metric] = np.std(all_metrics[metric])

final_value_per_metric = {}
final_std_per_metric = {}
for metric in ["real_mae", "real_rmse", "r2", "correlation"]:
    final_value_per_metric[metric] = average_metrics[metric]
    final_std_per_metric[metric] = std_metrics[metric]

with open(os.path.join(resdir, 'final_values.json'), 'w') as fp:
    json.dump(final_value_per_metric, fp)

with open(os.path.join(resdir, 'final_stds.json'), 'w') as fp:
    json.dump(final_std_per_metric, fp)