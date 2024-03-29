import argparse
import json
import os
import sys
import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error,
                             mean_absolute_error, balanced_accuracy_score,
                             roc_auc_score)
from sklearn.preprocessing import (KBinsDiscretizer, StandardScaler, 
                                   OrdinalEncoder)
from sklearn.linear_model import Ridge, LogisticRegression
import pandas as pd
import torch
from torch import nn, optim
from torchvision import transforms
from surfify.models import SphericalHemiFusionEncoder
from surfify.utils import (setup_logging, icosahedron, downsample_data,
                           downsample)

from datasets import DataManager
from augmentations import Normalize, Reshape, Transformer
from utils import params_from_args, encoder_cp_from_model_cp


# Get user parameters
parser = argparse.ArgumentParser(description="Spherical predictor")
parser.add_argument(
    "--data", default="openbhb", choices=("hbn", "openbhb"),
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
    "--setups-file", default="None", required=True,
    help="the path to the file linking the setups to the trained encoder's path.")
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
all_modalities = modalities.copy()
metrics = ["thickness", "curv", "sulc"]
n_features = len(metrics)
transform = None
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            if (getattr(args, arg_name, None) is None or not
                relation_mapper[value[0]](getattr(args, arg_name), value[1])):
                matches = False
        if matches:
            return case
    return None

setups = pd.read_table(args.setups_file)

validation_metric = "mae" if args.method == "regression" else "bacc"
best_is_low = validation_metric == "mae"
regressor_params = [0.01, 0.1, 1, 10, 100]
def cases(latent_dim=128, batch_size=1024):
    cases = dict(
        simCLR_base={"algo": "simCLR", "hemimixup": 0, "groupmixup": 0, "cutout": True, "blur": True,
        "noise": True, "latent_dim": latent_dim, "batch_size": batch_size},
        simCLR_base_hemi={"algo": "simCLR", "hemimixup": ["gt", 0], "groupmixup": 0, "cutout": True,
        "blur": True, "noise": True, "latent_dim": latent_dim, "batch_size": batch_size},
        simCLR_base_group={"algo": "simCLR", "hemimixup": 0, "groupmixup": ["gt", 0], "cutout": True,
        "blur": True, "noise": True, "latent_dim": latent_dim, "batch_size": batch_size},
        sex_supervised={"supervised": True, "latent_dim": latent_dim, "batch_size": batch_size, "to_predict": "sex"},
        age_supervised={"supervised": True, "latent_dim": latent_dim, "batch_size": batch_size, "to_predict": "age"})
    return cases
cases = cases(128, 1024)
best_cp_per_case = dict()
to_predict, method = args.to_predict, args.method
for setup_id in setups["id"].values:
    params, epoch, best_param, best_value = setups[
        setups["id"] == setup_id][[
            "args", f"best_epoch_{args.to_predict}",
            f"best_param_{args.to_predict}",
            f"best_value_{args.to_predict}"]].values[0]
    compute_stds = False
    same_params_setups = setups[setups["args"] == params]
    if len(same_params_setups) > 1:
        compute_stds = True
    to_predict = args.to_predict
    local_args, supervised = params_from_args(params, args)
    if not hasattr(local_args, "algo"):
        local_args.algo = "barlow"
    if not hasattr(local_args, "sigma"):
        local_args.sigma = 0
    if not hasattr(local_args, "noise"):
        local_args.noise = False
    if not hasattr(local_args, "projector"):
        local_args.projector = "256-512-512" if local_args.algo == "barlow" else "256-128"
    if supervised:
        local_args.supervised = True
    case = matching_args_to_case(local_args, cases)
    args.to_predict = to_predict
    if case is not None and epoch > 0 and local_args.ico_order == 5:
        if compute_stds:
            metric_per_epoch_per_param = [[[] for _ in range(int(local_args.epochs / 10))] for _ in regressor_params]
            setup_ids = []
            for _setup_id in same_params_setups["id"].values:
                cp_path = os.path.join(
                    "/".join(args.setups_file.split("/")[:-1]),
                    "checkpoints", str(_setup_id))
                validation_metrics_path = os.path.join(
                    cp_path, f"validation_metrics_{args.to_predict}.json")
                if not os.path.exists(validation_metrics_path):
                    continue
                with open(validation_metrics_path, "r") as f:
                    validation_metrics = json.load(f)
                for param_idx, _ in enumerate(regressor_params):
                    for epoch_idx, epoch in enumerate(validation_metrics["epochs"]):
                        nth_cp = int(epoch / 10) - 1
                        metric_per_epoch_per_param[param_idx][
                            nth_cp].append(validation_metrics[
                                validation_metric][param_idx][epoch_idx])
                setup_ids.append(_setup_id)
            if len(setup_ids) == 0:
                continue
            average_metric_per_epoch_per_param = np.mean(
                metric_per_epoch_per_param, axis=2)
            best_epoch_per_param_index = np.argsort(
                average_metric_per_epoch_per_param)[
                    :, 0 if best_is_low else -1]
            best_param_idx = np.argsort(average_metric_per_epoch_per_param[
                list(range(len(regressor_params))), best_epoch_per_param_index])[
                    0 if best_is_low else -1]
            best_value = average_metric_per_epoch_per_param[
                best_param_idx][best_epoch_per_param_index[best_param_idx]]
            best_param = regressor_params[best_param_idx]
            best_epoch = int((best_epoch_per_param_index[
                best_param_idx] + 1) * 10)
            checkpoints = []
            for _setup_id in setup_ids:
                cp_path = os.path.join(
                    "/".join(args.setups_file.split("/")[:-1]),
                    "checkpoints", str(_setup_id),
                    f"model_epoch_{best_epoch}.pth")
                checkpoint = torch.load(cp_path, map_location=device)
                encoder_prefix = "0" if supervised else "backbone"
                checkpoint = encoder_cp_from_model_cp(checkpoint, encoder_prefix)
                checkpoints.append(checkpoint)
            if case not in best_cp_per_case.keys():
                best_cp_per_case[case] = (
                    setup_ids, checkpoints, best_value, best_param, best_epoch)
            if ((best_cp_per_case[case][2] > best_value and best_is_low
                    and len(best_cp_per_case[case][0]) <= len(setup_ids))
                or (best_cp_per_case[case][2] < best_value and
                    not best_is_low and
                    len(best_cp_per_case[case][0]) <= len(setup_ids))
                    or len(best_cp_per_case[case][0]) < len(setup_ids)):
                best_cp_per_case[case] = (
                    setup_ids, checkpoints, best_value, best_param, best_epoch)
        else:
            pretrained_path = os.path.join(
                "/".join(args.setups_file.split("/")[:-1]),
                "checkpoints", str(setup_id),
                "model_epoch_{}.pth".format(int(epoch)))
            if int(setup_id) < 10000:
                pretrained_path = params
                pretrained_path = pretrained_path.replace(
                    "encoder.pth", "model_epoch_{}.pth".format(int(epoch)))
            checkpoint = torch.load(pretrained_path, map_location=device)
            encoder_prefix = "0" if supervised else "backbone"
            checkpoint = encoder_cp_from_model_cp(checkpoint, encoder_prefix)
            
            if case not in best_cp_per_case.keys():
                best_cp_per_case[case] = (
                    [setup_id], [checkpoint], best_value, best_param, epoch)
            if ((best_cp_per_case[case][2] > best_value and best_is_low)
                or (best_cp_per_case[case][2] < best_value and
                    not best_is_low)) and len(best_cp_per_case[case][0]) <= 1:
                best_cp_per_case[case] = (
                    [setup_id], [checkpoint], best_value, best_param, epoch)
checkpoints = [best_cp_per_case[case][1] for case in cases.keys()
                if case in best_cp_per_case.keys()]
setup_ids = [best_cp_per_case[case][0] for case in cases.keys()
                if case in best_cp_per_case.keys()]
best_regressor_params = [best_cp_per_case[case][3] for case in cases.keys()
                            if case in best_cp_per_case.keys()]
cases_names = [case for case in cases.keys()
                if case in best_cp_per_case.keys()]
args.to_predict, args.method = to_predict, method

# print(setup_ids)
print("".join(f"{case} : {best_cp_per_case[case][0]}, best value "
      f"{best_cp_per_case[case][2]} with {best_cp_per_case[case][3]} at "
      f"epoch {best_cp_per_case[case][4]}.\n"
      for case in best_cp_per_case.keys()))
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

transform = {"surface-lh": transform,
             "surface-rh": transform}

kwargs = {
    "surface-rh": {"metrics": metrics},
    "surface-lh": {"metrics": metrics},
}

if args.data == "hbn":
    kwargs["surface-lh"]["symetrized"] = True
    kwargs["surface-rh"]["symetrized"] = True
    if args.to_predict not in ["sex", "age", "site"]:
        all_modalities.append("clinical")
        kwargs["clinical"] = dict(cols=[args.to_predict])

test_size = "defaults"
stratify = ["sex", "age", "site"]
validation = None
if args.data != "openbhb":
    test_size = 0.2
    validation = 5


dataset = DataManager(
    dataset=args.data, datasetdir=args.datadir, modalities=all_modalities,
    stratify=stratify, discretize=["age"], transform=transform,
    overwrite=False, test_size=test_size, validation=validation,
    **kwargs)

params_for_validation = {
    "regression": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "classification": {"C": [0.01, 0.1, 1, 10, 100]}
}

best_params = {
    "regression": {"alpha": 1},
    "classification": {"C": 1, "max_iter": 10000}
}

if validation is not None:
    loaders = []
    for fold in range(validation):
        loaders.append(torch.utils.data.DataLoader(
            dataset["train"][fold]["train"], batch_size=batch_size, num_workers=6,
            pin_memory=True, shuffle=True))
    loaders.append(torch.utils.data.DataLoader(
            dataset["train"]["all"], batch_size=batch_size, num_workers=6,
            pin_memory=True, shuffle=True))
else:
    loaders = [torch.utils.data.DataLoader(
        dataset["train"], batch_size=batch_size, num_workers=6,
        pin_memory=True, shuffle=True)]


all_label_data = []
if "clinical" in all_modalities:
    clinical_names = np.load(
        os.path.join(args.datadir, "clinical_names.npy"), allow_pickle=True)

for fold_idx, loader in enumerate(loaders):
    X = {mod: [] for mod in modalities}
    for data in loader:
        data, metadata, _ = data
        if args.to_predict in metadata.keys() and fold_idx == len(loaders) - 1:
            all_label_data.append(metadata[args.to_predict])
        elif fold_idx == len(loaders) - 1:
            index_to_predict = clinical_names.tolist().index(args.to_predict)
            all_label_data.append(data["clinical"][:, index_to_predict])
        for modality in modalities:
            suffix = f"_fold_{fold_idx}"
            if fold_idx == len(loaders) - 1:
                suffix = ""
            path_to_scaler = os.path.join(
                args.datadir, f"{modality}_scaler{suffix}.save")
            if not os.path.exists(path_to_scaler):
                X[modality] += data[modality].view(
                    (len(data[modality]), -1)).tolist()
    for modality in modalities:
        suffix = f"_fold_{fold_idx}"
        if fold_idx == len(loaders) - 1:
            suffix = ""
        path_to_scaler = os.path.join(
            args.datadir, f"{modality}_scaler{suffix}.save")
        if not os.path.exists(path_to_scaler):
            print("Fit scaler")
            scaler = StandardScaler()
            scaler.fit(X[modality])
            joblib.dump(scaler, path_to_scaler)


all_label_data = np.concatenate(all_label_data)
label_values = np.unique(all_label_data)
# plt.hist(all_label_data)

def corr_metric(y_true, y_pred):
    mat = np.concatenate([y_true[:, np.newaxis], y_pred[:, np.newaxis]], axis=1)
    corr_mat = np.corrcoef(mat, rowvar=False)
    return corr_mat[0, 1]

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

output_activation = nn.Identity()
hidden_dim = 128
tensor_type = "float"
out_to_pred_func = lambda x: x
out_to_real_pred_func = lambda x: x
root_mean_squared_error = lambda x, y: mean_squared_error(x, y, squared=False)
evaluation_against_real_metric = {
    "mae": mean_absolute_error,
    "r2": r2_score, "correlation": corr_metric}
if args.method == "regression":
    output_dim = 1
    
    label_prepro = StandardScaler()
    label_prepro.fit(all_label_data[:, np.newaxis])
    
    evaluation_metrics = {}#"mae": mean_absolute_error}
    regressor = Ridge
    out_to_real_pred_func = lambda x: label_prepro.inverse_transform(x[:, np.newaxis]).squeeze()
else:
    output_dim = len(label_values)
    evaluation_metrics = {"accuracy": accuracy_score,
                          "bacc": balanced_accuracy_score,
                          "auc": roc_auc_score}
    evaluation_against_real_metric = {}
    validation_metric = "auc"
    tensor_type = "long"
    label_prepro = TransparentProcessor()
    out_to_real_pred_func = lambda x : x.squeeze()
    n_bins = 3
    if any([type(value) is np.str_ for value in label_values]):
        label_prepro = OrdinalEncoder()
        label_prepro.fit(all_label_data[:, np.newaxis])
        out_to_real_pred_func = lambda x : label_prepro.inverse_transform(
            x[:, np.newaxis]).squeeze()
    elif output_dim > n_bins:
        label_prepro = KBinsDiscretizer(n_bins=n_bins, encode="ordinal")
        label_prepro.fit(all_label_data[:, np.newaxis])
        print(label_prepro.bin_edges_)
        validation_metric = "bacc"
        evaluation_metrics = {"bacc": balanced_accuracy_score}
        evaluation_against_real_metric = {
            "mae": mean_absolute_error, "r2": r2_score,
            "correlation": corr_metric}
        out_to_real_pred_func = lambda x : label_prepro.inverse_transform(
            x[:, np.newaxis]).squeeze()
    best_is_low = False
    out_to_pred_func = lambda x: x.squeeze()
    regressor = LogisticRegression

for case_id, (setup_id, checkpoint) in enumerate(zip(setup_ids, checkpoints)):
    print(f"Best setup for case {cases_names[case_id]} : {setup_id}")
    case_final_perfs = {name: [] for name in (
        list(evaluation_metrics) + list(evaluation_against_real_metric))}
    for setup_idx, _setup_id in enumerate(setup_id):
        params = setups[setups["id"] == _setup_id]["args"].values[0]
        to_predict, method = args.to_predict, args.method
        local_args, supervised = params_from_args(params, args)
        args.to_predict, args.method = to_predict, method
        best_params["regression"]["alpha"] = best_regressor_params[case_id] 

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

        # if scaling creating scalers for train, test and if there are valid folds
        scaling = local_args.standardize
        if scaling and validation is None:
            scalers = {mod: None for mod in modalities}
            for modality in modalities:
                datadir = args.datadir
                path_to_scaler = os.path.join(
                    datadir, f"{modality}_scaler.save")
                scaler = joblib.load(path_to_scaler)
                scalers[modality] =  transforms.Compose([
                    Reshape((1, -1)),
                    scaler.transform,
                    transforms.ToTensor(),
                    torch.squeeze,
                    Reshape(input_shape),
                ])
        elif scaling:
            scalers = dict(train=[])
            
            for fold in range(validation):
                scalers["train"].append({mod: None for mod in modalities})
                for modality in modalities:
                    datadir = args.datadir
                    path_to_scaler = os.path.join(
                        datadir, f"{modality}_scaler_fold_{fold}.save")
                    scaler = joblib.load(path_to_scaler)
                    scalers["train"][fold][modality] =  transforms.Compose([
                        Reshape((1, -1)),
                        scaler.transform,
                        transforms.ToTensor(),
                        torch.squeeze,
                        Reshape(input_shape),
                    ])
            scalers["train"].append({mod: None for mod in modalities})
            scalers["test"] = {mod: None for mod in modalities}
            for modality in modalities:
                datadir = args.datadir
                path_to_scaler = os.path.join(
                    datadir, f"{modality}_scaler.save")
                scaler = joblib.load(path_to_scaler)
                scaler_transform =  transforms.Compose([
                    Reshape((1, -1)),
                    scaler.transform,
                    transforms.ToTensor(),
                    torch.squeeze,
                    Reshape(input_shape),
                ])
                scalers["train"][-1][modality] = scaler_transform
                scalers["test"][modality] = scaler_transform
    
        # Creating on_the_fly transform for train, test and possibly valid folds
        if validation is None:
            on_the_fly_transform = dict()
            for modality in modalities:
                transformer = Transformer()
                if scaling:
                    transformer.register(scalers[modality])
                if local_args.normalize:
                    transformer.register(Normalize())
                on_the_fly_transform[modality] = transformer
        else:
            on_the_fly_transform = dict(train=[], test=dict())
            for fold in range(validation):
                on_the_fly_transform["train"].append(dict())
                for modality in modalities:
                    transformer = Transformer()
                    if scaling:
                        transformer.register(scalers["train"][fold][modality])
                    if local_args.normalize:
                        transformer.register(Normalize())
                    on_the_fly_transform["train"][fold][modality] = transformer
            on_the_fly_transform["train"].append(dict())
            for modality in modalities:
                transformer = Transformer()
                if scaling:
                    transformer.register(scalers["train"][-1][modality])
                if local_args.normalize:
                    transformer.register(Normalize())
                on_the_fly_transform["train"][-1][modality] = transformer
                on_the_fly_transform["test"][modality] = transformer
        
        dataset = DataManager(
            dataset=args.data, datasetdir=args.datadir, modalities=all_modalities,
            stratify=stratify, discretize=["age"], validation=validation,
            transform=transform, on_the_fly_transform=on_the_fly_transform,
            overwrite=False, test_size=test_size, **kwargs)
        if args.data == "openbhb":
            dataset.create_val_from_test(
                val_size=0.5, stratify=["sex", "age", "site"], discretize=["age"])
    
        all_metrics = {}
        for name in evaluation_metrics.keys():
            all_metrics[name] = []
        for name in evaluation_against_real_metric.keys():
            all_metrics[name] = []

        params = params_for_validation[args.method]
        param_name = list(params)[0]
        for value_idx, value in enumerate(params[param_name]):
            for name in all_metrics.keys():
                all_metrics[name].append([])

        run_name = (
            "evaluate_representations_on_{}_to_predict_{}_via_{}_for_setup_{}"
            "_case_{}").format(
                args.data, args.to_predict, args.method, setup_id,
                list(cases)[case_id])

        case_resdir = os.path.join(resdir, run_name)
        if not os.path.isdir(case_resdir):
            os.makedirs(case_resdir)

        valid_loaders = []
        if validation is None:
            train_loaders = [torch.utils.data.DataLoader(
                dataset["train"], batch_size=batch_size, num_workers=6,
                pin_memory=True, shuffle=True)]
        else:
            train_loaders = []
            for fold in range(validation):
                train_loaders.append(torch.utils.data.DataLoader(
                    dataset["train"][fold]["train"], batch_size=batch_size,
                    num_workers=6, pin_memory=True, shuffle=True))
                valid_loaders.append(torch.utils.data.DataLoader(
                    dataset["train"][fold]["valid"], batch_size=batch_size,
                    num_workers=6, pin_memory=True, shuffle=True))
            train_loaders.append(torch.utils.data.DataLoader(
                dataset["train"]["all"], batch_size=batch_size, num_workers=6,
                pin_memory=True, shuffle=True))
        test_dataset = dataset["test"]
        if local_args.data_train == args.data:
            test_dataset = test_dataset["test"]
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, num_workers=6,
            pin_memory=True, shuffle=True)

        encoder.load_state_dict(checkpoint[setup_idx])


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = encoder.to(device)

        model.eval()
        for fold, (train_loader, test_loader) in enumerate(
            zip(train_loaders, valid_loaders + [test_loader])):
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
                new_y = label_prepro.transform(np.array(y)[:, np.newaxis]).squeeze()
                transformed_ys.append(new_y)
                ys.append(y)
                with torch.cuda.amp.autocast():
                    data = (left_x, right_x)
                    latents.append(model(data).squeeze().detach().cpu().numpy())
            Y = np.concatenate(transformed_ys)
            real_y = np.concatenate(ys)
            X = np.concatenate(latents)
            
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
                new_y = label_prepro.transform(np.array(y)[:, np.newaxis]).squeeze()
                test_ys.append(y)
                test_transformed_ys.append(new_y)
                with torch.cuda.amp.autocast():
                    data = (left_x, right_x)
                    test_latents.append(model(data).squeeze().detach().cpu().numpy())
            X_test = np.concatenate(test_latents)
            Y_test = np.concatenate(test_transformed_ys)
            real_y_test = np.concatenate(test_ys)

            if fold < len(train_loaders) - 1:
                params = params_for_validation[args.method]
                param_name = list(params)[0]
                for value_idx, value in enumerate(params[param_name]):
                    local_params = {param_name: value}
                    if args.method == "classification":
                        local_params["max_iter"] = 10000
                    local_regressor = regressor(**local_params)
                    local_regressor.fit(X, Y)
                    y_hat = local_regressor.predict(X_test).squeeze()

                    preds = out_to_pred_func(y_hat)
                    real_preds = out_to_real_pred_func(y_hat)

                    for name, metric in evaluation_metrics.items():
                        all_metrics[name][value_idx].append(metric(Y_test, preds))
                    for name, metric in evaluation_against_real_metric.items():
                        all_metrics[name][value_idx].append(metric(real_y_test, real_preds))
            elif len(train_loaders) > 1:
                for value_idx, value in enumerate(params[param_name]):
                    for metric in all_metrics.keys():
                        all_metrics[metric][value_idx] = np.mean(all_metrics[metric][value_idx])
                sorted_index = np.argsort(all_metrics[validation_metric])
                best_value = np.array(params[param_name])[sorted_index][0 if best_is_low else -1]
                best_params[args.method][param_name] = best_value
            if fold == len(train_loaders) - 1:
                local_regressor = regressor(**best_params[args.method])
                local_regressor.fit(X, Y)
                y_hat = local_regressor.predict(X_test).squeeze()

                preds = out_to_pred_func(y_hat)
                real_preds = out_to_real_pred_func(y_hat)
                
                final_value_per_metric = {}
                for name, metric in evaluation_metrics.items():
                    final_value_per_metric[name] = metric(Y_test, preds)
                for name, metric in evaluation_against_real_metric.items():
                    final_value_per_metric[name] = metric(real_y_test, real_preds)


        for metric in final_value_per_metric.keys():
            final_value = final_value_per_metric[metric]
            # print(final_value)
            case_final_perfs[metric].append(final_value)
        with open(os.path.join(resdir, 'final_values.json'), 'w') as fp:
            json.dump(final_value_per_metric, fp)
    for metric in case_final_perfs.keys():
        final_value = np.mean(case_final_perfs[metric])
        final_std = np.std(case_final_perfs[metric])
        print(f"Test {metric} {final_value} +- {final_std}")
        
