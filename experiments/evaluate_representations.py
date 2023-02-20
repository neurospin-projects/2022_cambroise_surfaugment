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
    "--setups-file", default="None", required=True,
    help="the path to the file linking the setups to the pretrained encoder's path.")
args = parser.parse_args()
args.conv = "DiNe"
args.ico_order = 5

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
n_features = len(metrics)
transform = None
batch_size = 32


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
        "batch_augment", "inter_modal_augment", "blur",
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
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        if value in ["True", "False"]:
            value = value == "True"
        setattr(args, args_names[idx], value)
    return args

def matching_args_to_case(args, cases):
    relation_mapper = {
        "gt": lambda x, y: x > y,
        "lt": lambda x, y: x < y,
        "gte": lambda x, y: x >= y,
        "lte": lambda x, y: x <= y,
        "eq": lambda x, y: x == y,
        "in": lambda x, y: x.isin(y),
        "out": lambda x, y: ~x.isin(y),
    }
    for case, case_args in cases.items():
        matches = True
        for arg_name, value in case_args.items():
            if not type(value) is list:
                value = ["eq", value]
            if not relation_mapper[value[0]](getattr(args, arg_name), value[1]):
                matches = False
        if matches:
            return case
    return None

setups = pd.read_table(args.setups_file)
if args.pretrained_setup != "None":
    params, epoch = setups[setups["id"] == int(args.pretrained_setup)][["args", "best_epoch"]].values[0]
    pretrained_path = os.path.join(
            "/".join(args.setups_file.split("/")[:-1]),
            "checkpoints", args.pretrained_setup,
            "model_epoch_{}.pth".format(int(epoch)))
    if int(args.pretrained_setup) < 10000:
        pretrained_path = params
        pretrained_path = pretrained_path.replace(
            "encoder.pth", "model_epoch_{}.pth".format(int(epoch)))
    args = params_from_args(params)
    
    checkpoint = torch.load(pretrained_path)
    checkpoint = encoder_cp_from_model_cp(checkpoint)
    checkpoints = [checkpoint]
    setup_ids = [int(args.pretrained_setup)]
else:
    validation_metric = "mae"
    best_is_low = True
    cases = dict(
        a={"algo": "simCLR", "inter_modal_augment": 0, "batch_augment": 0, "cutout": True, "blur": True,
           "noise": True, "sigma": 0},
        b={"algo": "simCLR", "inter_modal_augment": ["gt", 0], "batch_augment": 0, "cutout": True,
           "blur": True, "noise": True, "sigma": 0},
        c={"algo": "simCLR", "inter_modal_augment": 0, "batch_augment": ["gt", 0], "cutout": True,
           "blur": True, "noise": True, "sigma": 0},
        d={"algo": "barlow", "inter_modal_augment": 0, "batch_augment": 0, "cutout": True, "blur": True,
           "noise": True},
        e={"algo": "barlow", "inter_modal_augment": ["gt", 0], "batch_augment": 0, "cutout": True,
           "blur": True, "noise": True},
        f={"algo": "barlow", "inter_modal_augment": 0, "batch_augment": ["gt", 0], "cutout": True,
           "blur": True, "noise": True})
    best_cp_per_case = dict()
    for setup_id in setups["id"].values:
        params, epoch = setups[setups["id"] == setup_id][["args", "best_epoch"]].values[0]
        local_args = params_from_args(params)
        if not hasattr(local_args, "algo"):
            local_args.algo = "barlow"
        if not hasattr(local_args, "sigma"):
            local_args.sigma = 0
        if not hasattr(local_args, "noise"):
            local_args.noise = False
        case = matching_args_to_case(local_args, cases)
        if case is not None and epoch > 0 and local_args.ico_order == 5:
            pretrained_path = os.path.join(
                "/".join(args.setups_file.split("/")[:-1]),
                "checkpoints", str(setup_id),
                "model_epoch_{}.pth".format(int(epoch)))
            if int(setup_id) < 10000:
                pretrained_path = params
                pretrained_path = pretrained_path.replace(
                    "encoder.pth", "model_epoch_{}.pth".format(int(epoch)))            
            checkpoint = torch.load(pretrained_path)
            checkpoint = encoder_cp_from_model_cp(checkpoint)
            validation_metrics_path = os.path.join(
                "/".join(pretrained_path.split("/")[:-1]),
                "validation_metrics.json")
            if os.path.exists(validation_metrics_path):
                with open(validation_metrics_path, "r") as f:
                    validation_metrics = json.load(f)
                best_metric = validation_metrics[validation_metric][
                    validation_metrics["epochs"].index(epoch)]
                if case not in best_cp_per_case.keys():
                    best_cp_per_case[case] = (
                        setup_id, checkpoint, best_metric)
                if ((best_cp_per_case[case][2] > best_metric and best_is_low)
                    or (best_cp_per_case[case][2] < best_metric and
                        not best_is_low)):
                    best_cp_per_case[case] = (
                        setup_id, checkpoint, best_metric)
    checkpoints = [best_cp_per_case[case][1] for case in cases.keys()
                   if case in best_cp_per_case.keys()]
    setup_ids = [best_cp_per_case[case][0] for case in cases.keys()
                 if case in best_cp_per_case.keys()]
    cases_names = [case for case in cases.keys() if case in best_cp_per_case.keys()]

print(setup_ids)
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
    overwrite=False, test_size="defaults", **kwargs)
if args.data_train == args.data:
    dataset.create_val_from_test(
        val_size=0.5, stratify=["sex", "age", "site"], discretize=["age"])


loader = torch.utils.data.DataLoader(
    dataset["train"], batch_size=batch_size, num_workers=6, pin_memory=True,
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

for case_id, (setup_id, checkpoint) in enumerate(zip(setup_ids, checkpoints)):
    print(f"Best setup for case {cases_names[case_id]} : {setup_id}")
    params = setups[setups["id"] == setup_id]["args"].values[0]
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
    if args.data_train == args.data:
        dataset.create_val_from_test(
            val_size=0.5, stratify=["sex", "age", "site"], discretize=["age"])
    all_metrics = {}
    for name in evaluation_metrics.keys():
        all_metrics[name] = []
    for name in evaluation_against_real_metric.keys():
        all_metrics[name] = []

    run_name = (
        "deepint_evaluate_representations_to_predict_{}_{}_predict_via_{}"
        "_pretrained_{}_case_{}").format(
            args.data, args.to_predict, args.method, setup_id,
            None if args.pretrained_setup != "None" else list(cases)[case_id])

    resdir = os.path.join(resdir, run_name)
    if not os.path.isdir(resdir):
        os.makedirs(resdir)

    train_loader = torch.utils.data.DataLoader(
        dataset["train"], batch_size=batch_size, num_workers=6, pin_memory=True,
        shuffle=True)

    test_dataset = dataset["test"]
    if args.data_train == args.data:
        test_dataset = test_dataset["test"]
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=6, pin_memory=True,
        shuffle=True)

    encoder.load_state_dict(checkpoint)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = encoder.to(device)

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
    # for name, metric in evaluation_against_real_metric.items():
    #     print(name, metric(real_y, real_preds))

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
    print(final_value_per_metric["real_mae"])
    with open(os.path.join(resdir, 'final_values.json'), 'w') as fp:
        json.dump(final_value_per_metric, fp)

    with open(os.path.join(resdir, 'final_stds.json'), 'w') as fp:
        json.dump(final_std_per_metric, fp)