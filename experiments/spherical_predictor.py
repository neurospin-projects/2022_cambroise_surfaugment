import argparse
import json
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import torch
from torch import nn, optim
from torchvision import transforms
from surfify.models import SphericalHemiFusionEncoder
from surfify.utils import setup_logging, icosahedron, downsample_data, downsample
from brainboard import Board

from multimodaldatasets.datasets import DataManager, DataLoaderWithBatchAugmentation
from augmentations import Permute, Normalize, GaussianBlur, RescaleAsImage, PermuteBeetweenModalities, Bootstrapping, Reshape, Transformer
from utils import params_from_args


# Get user parameters
parser = argparse.ArgumentParser(description="Spherical predictor")
parser.add_argument(
    "--data", default="hcp", choices=("hbn", "euaims", "hcp", "openbhb", "privatebhb"),
    help="the input cohort name.")
parser.add_argument(
    "--datadir", metavar="DIR", help="data directory path.", required=True)
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
    "--conv-filters", default="128-128-256-256", type=str, metavar="F",
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
    "--evaluate", action="store_true",
    help="Uses the whole training set to train once, and evaluates on the hold-out test set."
)
parser.add_argument(
    "--batch-augment", "-ba", default=0.0, type=float,
    help="optionnally uses batch augmentation.")
parser.add_argument(
    "--inter-modal-augment", "-ima", default=0.0, type=float,
    help="optionnally uses inter modality augment.")
args = parser.parse_args()
args.ngpus_per_node = torch.cuda.device_count()
args.ico_order = 5

# Prepare process
run_id = int(time.time())
setup_logging(level="info", logfile=None)
checkpoint_dir = os.path.join(args.outdir, "predict_{}".format(args.to_predict))
checkpoint_dir = os.path.join(checkpoint_dir, str(run_id))
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(os.path.join(checkpoint_dir, "params.json"), "w") as file:
    json.dump(vars(args), file)
checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
stats_file = open(os.path.join(checkpoint_dir, "stats.txt"), "a", buffering=1)
print(" ".join(sys.argv))
print(" ".join(sys.argv), file=stats_file)

# Load the input cortical data
modalities = ["surface-lh", "surface-rh"]
metrics = ["thickness", "curv", "sulc"]
args.conv = "DiNe"
transform = None

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
    name_to_check = "backbone"
    checkpoint = {".".join(key.split(".")[1:]): value 
                for key, value in checkpoint["model_state_dict"].items() if name_to_check in key}
    return checkpoint

checkpoint = None
if args.pretrained_setup != "None":
    assert args.setups_file != "None"
    setups = pd.read_table(args.setups_file)
    params, epoch = setups[setups["id"] == int(args.pretrained_setup)][["args", "best_epoch"]].values[0]
    pretrained_path = os.path.join(
        "/".join(args.setups_file.split("/")[:-1]),
        "checkpoints", args.pretrained_setup,
        "model_epoch_{}.pth".format(int(epoch)))
    if int(args.pretrained_setup) < 10000:
        pretrained_path = params
        pretrained_path = pretrained_path.replace(
            "encoder.pth", "model_epoch_{}.pth".format(int(epoch)))
    epochs, lr = args.epochs, args.learning_rate
    args = params_from_args(params, args)
    args.epochs, args.learning_rate = epochs, lr
    checkpoint = torch.load(pretrained_path)
    checkpoint = encoder_cp_from_barlow_cp(checkpoint)
# else:
#     args.n_features = len(metrics)
#     args.fusion_level = 1
#     args.activation = "ReLU"
#     args.standardize = True
#     args.normalize = False
#     args.batch_norm = False
#     args.conv_filters = [128, 128, 256, 256]
#     args.gaussian_blur_augment = False
#     args.cutout = False
args.conv_filters = [int(item) for item in args.conv_filters.split("-")]

input_shape = (len(metrics), len(ico_verts))

scaling = True
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
normalize = True
on_the_fly_transform = dict()
for modality in modalities:
    transformer = Transformer()
    if scaling:
        transformer.register(scalers[modality])
    if normalize:
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
test_size = None
kwargs = {
    "surface-rh": {"metrics": metrics,
                   "z_score": False},
    "surface-lh": {"metrics": metrics,
                   "z_score": False},
    # "test_size": 0.2
}

stratify = ["sex", "age", "site"]
if args.to_predict == "asd":
    stratify.append("asd")

if args.data in ["hbn", "euaims"]:
    kwargs["surface-lh"]["symetrized"] = True

    kwargs["surface-rh"]["symetrized"] = True

    if args.to_predict not in stratify:
        modalities.append("clinical")
    test_size = 0.2

if args.data == "openbhb":
    kwargs["test_size"] = None

n_folds = None

dataset = DataManager(dataset=args.data, datasetdir=args.datadir,
                      modalities=modalities, validation=n_folds,
                      stratify=stratify, discretize=["age"],
                      transform=transform, on_the_fly_transform=on_the_fly_transform,
                      overwrite=False, test_size=test_size, **kwargs)
dataset.create_val_from_test(
    val_size=0.5, stratify=stratify, discretize=["age"])


loader = torch.utils.data.DataLoader(
    dataset["train"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    dataset["test"]["valid"], batch_size=args.batch_size, num_workers=6,
    pin_memory=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset["test"]["test"], batch_size=args.batch_size, num_workers=6,
    pin_memory=True, shuffle=True)

class TransparentProcessor(object):
    def __init__(self):
        super().__init__()
    
    def transform(self, x):
        if type(x) is torch.Tensor:
            return x.cpu().detach().numpy()
        return np.asarray(x)
    
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
elif args.method == "classification":

    output_dim = len(label_values)
    evaluation_metrics = {"accuracy": accuracy_score, "bacc": balanced_accuracy_score,
                          "auc": roc_auc_score}
    tensor_type = "long"
    n_bins = 3
    label_prepro = TransparentProcessor()
    output_activation = nn.Softmax(dim=1)
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

im_shape = next(iter(loader))[0]["surface-lh"][0].shape
n_features = im_shape[0]


activation = "ReLU"
all_metrics = {}
for name in evaluation_metrics.keys():
    all_metrics[name] = [[] for _ in range(args.epochs)]
for name in evaluation_against_real_metric.keys():
    all_metrics[name] = [[] for _ in range(args.epochs)]

use_board = True
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


# for fold in range(n_folds):
#     print("Training on fold {} / {}".format(fold + 1, n_folds))
    # if args.batch_augment > 0:
    #     original_dataset = DataManager(
    #         dataset=args.data, datasetdir=args.datadir,
    #         modalities=modalities, transform=transform,
    #         stratify=["sex", "age"], discretize=["age"],
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

    # train_loader = DataLoaderWithBatchAugmentation(batch_transforms,
    #     dataset["train"][fold]["train"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #     shuffle=True)
    # valid_loader = DataLoaderWithBatchAugmentation(batch_transforms_valid,
    #     dataset["train"][fold]["valid"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #     shuffle=True)


    # train_loader = torch.utils.data.DataLoader(
    #     dataset["train"][fold]["train"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #     shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(
    #     dataset["train"][fold]["valid"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #     shuffle=True)
    # if args.evaluate:
    #     train_loader = DataLoaderWithBatchAugmentation(batch_transforms,
    #         dataset["train"]["all"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #         shuffle=True)
    #     valid_loader = DataLoaderWithBatchAugmentation(batch_transforms_valid,
    #         dataset["test"], batch_size=args.batch_size, num_workers=6, pin_memory=True,
    #         shuffle=True)

encoder = SphericalHemiFusionEncoder(
    n_features, args.ico_order, args.latent_dim, fusion_level=args.fusion_level,
    conv_flts=args.conv_filters, activation=activation,
    batch_norm=args.batch_norm, conv_mode=args.conv,
    cachedir=os.path.join(args.outdir, "cached_ico_infos"))

if checkpoint is not None:
    encoder.load_state_dict(checkpoint)

all_encoder_params = list(encoder.parameters())
assert np.abs(args.freeze_up_to) < len(all_encoder_params)
idx_to_freeze = []
if args.freeze_up_to != 0:
    number_of_layers_per_layer = 2 if not args.batch_norm else 3
    if args.freeze_up_to < 0:
        args.freeze_up_to = len(all_encoder_params) - args.freeze_up_to
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

model = nn.Sequential(encoder, predictor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

    # print(model)
print("Number of trainable parameters : ",
    sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                    weight_decay=args.weight_decay)
# if args.momentum > 0:
#     optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
#                         weight_decay=args.weight_decay, momentum=args.momentum)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)
# scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.97 if epoch % 5 == 0 else 1)

if args.epochs > 0 and use_board:
    board = Board(env=str(run_id))

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
    with tqdm(total=len(loader), desc="Training epoch : {} / {}".format(epoch, args.epochs),
            postfix={"loss": 0, "lr": args.learning_rate, "average time": 0}, disable=not show_pbar) as pbar:
        for step, x in enumerate(loader):
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
            if args.to_predict == "asd":
                new_y -= 1
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                X = (left_x, right_x)
                y_hat = model(X).squeeze()
                loss = criterion(y_hat, new_y)
                preds = out_to_pred_func(y_hat)
                real_preds = out_to_real_pred_func(y_hat)

            scaler.scale(loss).backward()
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
            if args.to_predict == "asd":
                new_y -= 1
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    X = (left_x, right_x)
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
    # groups = {"closeness": ["real_mae", "real_rmse"], "coherence": ["r2", "correlation"]}
    # limit_per_group = {"closeness": (0, 10), "coherence": (0, 1)}
    # for name, group in groups.items():
    #     plt.figure()
    #     for metric in group:
    #         values = average_metrics[metric]
    #         plt.plot(range(args.epochs), values, label=metric)
    #     plt.title("Average validation metrics")
    #     plt.ylim(limit_per_group[name])
    #     plt.xlabel("epoch")
    #     plt.legend()
    #     plt.savefig(os.path.join(
    #         checkpoint_dir,"validation_metrics_{}.png".format(name)))

    best_epochs_per_metric = {}
    best_values_per_metric = {}
    best_stds_per_metric = {}
    the_lower_the_better = ["real_mae", "real_rmse"]
    for metric in all_metrics.keys():#["real_mae", "real_rmse", "r2", "correlation"]:
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
    