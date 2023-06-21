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
from sklearn.metrics import (
    balanced_accuracy_score, r2_score, mean_squared_error,
    mean_absolute_error, roc_auc_score)
from sklearn.preprocessing import (KBinsDiscretizer, StandardScaler,
                                   OrdinalEncoder)

from multimodaldatasets.datasets import DataManager
from surfify.utils import icosahedron, downsample, downsample_data
from surfify.models import SphericalHemiFusionEncoder
from augmentations import Normalize, Reshape, Transformer
from utils import params_from_args, encoder_cp_from_model_cp


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
all_modalities = modalities.copy()
metrics = ["thickness", "curv", "sulc"]
n_features = len(metrics)
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


setups = pd.read_table(args.setups_file)
if not f"best_epoch_{args.to_predict}" in setups.columns:
    setups[[
        f"best_epoch_{args.to_predict}",
        f"best_param_{args.to_predict}",
        f"best_value_{args.to_predict}"]] = np.nan
    setups.to_csv(args.setups_file, index=False, sep="\t")

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

transform = {
    "surface-rh": transform,
    "surface-lh": transform,
}


kwargs = {
    "surface-rh": {"metrics": metrics},
    "surface-lh": {"metrics": metrics},
}

if args.data in ["hbn", "euaims"]:
    kwargs["surface-lh"]["symetrized"] = True
    kwargs["surface-rh"]["symetrized"] = True
    if args.to_predict not in ["sex", "age", "site", "asd"]:
        all_modalities.append("clinical")
        kwargs["clinical"] = dict(cols=[args.to_predict])

# evaluation_metrics = (
#     {"mae": mean_absolute_error, "r2": r2_score} if args.method == "regression"
#     else {"auc": roc_auc_score, "bacc": balanced_accuracy_score})
# final_metric = "mae" if args.method == "regression" else "auc"
# what_is_best = {"mae": "lower", "r2": "higher", "bacc": "higher", "auc": "higher"}
# output_activation = nn.Identity()
# hidden_dim = 256
# tensor_type = "float"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# out_to_real_pred_func = []
# root_mean_squared_error = lambda x, y: mean_squared_error(x, y, squared=False)
# evaluation_against_real_metric = {"real_rmse": root_mean_squared_error, "real_mae": mean_absolute_error, "r2": r2_score, "correlation": corr_metric}
# label_prepro = lambda x: x.squeeze()

# if args.method == "classification":
#     output_dim = len(label_values)
#     evaluation_metrics = {"accuracy": accuracy_score, "bacc": balanced_accuracy_score,
#                           "auc": roc_auc_score}
#     tensor_type = "long"
#     # tensor_type = "float"
#     n_bins = 100
#     output_activation = nn.Softmax(dim=1)
#     # output_activation = nn.Sigmoid()
#     # output_activation = nn.Identity()
#     output_dim = 2
#     for idx in range(len(train_loaders)):
#         label_prepro.append(TransparentProcessor())
#         # out_to_real_pred_func.append(
#         #     lambda x : x.argmax(1).cpu().detach().numpy())
#         out_to_real_pred_func.append(
#             lambda x : x.round().cpu().detach().numpy())
#         if any([type(value) is np.str_ for value in label_values]):
#             label_prepro[idx] = OrdinalEncoder()
#             label_prepro[idx].fit(all_label_data[idx][:, np.newaxis])
#             print(label_prepro[idx].categories_)
#             out_to_real_pred_func[idx] = lambda x : label_prepro[idx].inverse_transform(
#                 x.argmax(1).cpu().detach().unsqueeze(1).numpy()).squeeze()
#         if output_dim > n_bins:
#             label_prepro[idx] = KBinsDiscretizer(n_bins=n_bins, encode="ordinal")
#             label_prepro[idx].fit(all_label_data[idx][:, np.newaxis])
#             print(label_prepro[idx].bin_edges_)
#             output_dim = n_bins
test_size = "defaults"
stratify = ["sex", "age", "site"]
validation = None
if args.data not in ["openbhb", "privatebhb"]:
    test_size = 0.2
    validation = 5
    if args.to_predict == "asd":
        stratify.append("asd")

dataset = DataManager(
    dataset=args.data, datasetdir=args.datadir, modalities=all_modalities,
    stratify=stratify, discretize=["age"], transform=transform,
    overwrite=False, test_size=test_size, validation=validation,
    **kwargs)

if "clinical" in all_modalities:
    clinical_names = np.load(
        os.path.join(args.datadir, "clinical_names.npy"), allow_pickle=True)
    if args.to_predict in clinical_names:
        index_to_predict = clinical_names.tolist().index(args.to_predict)

params_for_validation = {
    "regression": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "classification": {"C": [0.01, 0.1, 1, 10, 100]}
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
            if (not os.path.exists(path_to_scaler) and
                not args.data == "privatebhb"):
                X[modality] += data[modality].view(
                    (len(data[modality]), -1)).tolist()
    for modality in modalities:
        suffix = f"_fold_{fold_idx}"
        if fold_idx == len(loaders) - 1:
            suffix = ""
        path_to_scaler = os.path.join(
            args.datadir, f"{modality}_scaler{suffix}.save")
        if (not os.path.exists(path_to_scaler) and
            not args.data == "privatebhb"):
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
out_to_real_pred_func = lambda x: x
root_mean_squared_error = lambda x, y: mean_squared_error(x, y, squared=False)
evaluation_against_real_metric = {
    "mae": mean_absolute_error,
    "r2": r2_score, "correlation": corr_metric}
validation_metric = "mae"
if args.method == "regression":
    output_dim = 1
    
    label_prepro = StandardScaler()
    label_prepro.fit(all_label_data[:, np.newaxis])
    
    evaluation_metrics = {}#"mae": mean_absolute_error}
    regressor = Ridge
    out_to_real_pred_func = lambda x: label_prepro.inverse_transform(x[:, np.newaxis]).squeeze()
else:
    output_dim = len(label_values)
    evaluation_metrics = {#"accuracy": accuracy_score,
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
best_is_low = (args.method == "regression" and
               validation_metric in ["mae", "mse"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for setup_id in setups["id"].values:
    params = setups[setups["id"] == setup_id]["args"].values[0]
    cp_name = str(setup_id)
    checkpoints_path = os.path.join(
        "/".join(args.setups_file.split("/")[:-1]),
        "checkpoints", cp_name)
    if int(setup_id) < 10000:
        cp_name = params
        checkpoints_path = "/".join(cp_name.split("/")[:-1])
    to_predict, method = args.to_predict, args.method
    local_args, supervised = params_from_args(params, args)
    args.to_predict, args.method = to_predict, method
    conv_filters = [int(num) for num in local_args.conv_filters.split("-")]

    if not hasattr(local_args, "ico_order"):
        local_args.ico_order = len(conv_filters) + 1
    
    path_to_metrics = os.path.join(checkpoints_path, f"validation_metrics_{args.to_predict}.json")
    last_checkpoint = os.path.join(checkpoints_path,
                                   f"model_epoch_{local_args.epochs}.pth")
    # print(local_args.ico_order)
    # print(os.path.exists(path_to_metrics))
    # print(checkpoints_path)
    # print(last_checkpoint)
    if (local_args.ico_order != 5 #or os.path.exists(path_to_metrics)
        or not os.path.exists(checkpoints_path)
        or not os.path.exists(last_checkpoint)
        or (hasattr(local_args, "loss_param") and local_args.loss_param != 2)):
        continue

    encoder = SphericalHemiFusionEncoder(
        n_features, local_args.ico_order, local_args.latent_dim,
        fusion_level=local_args.fusion_level, conv_flts=conv_filters,
        activation=local_args.activation, batch_norm=local_args.batch_norm,
        conv_mode="DiNe",
        cachedir=os.path.join(args.outdir, "cached_ico_infos"))
    
    print("Number of trainable parameters : ",
        sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    
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

    
    param_name = "alpha" if args.method == "regression" else "C"
    regressor_params = params_for_validation[args.method][param_name]
    all_metrics = {}
    for name in evaluation_metrics.keys():
        all_metrics[name] = []
        for _ in regressor_params:
            all_metrics[name].append([])
    for name in evaluation_against_real_metric.keys():
        all_metrics[name] = []
        for _ in regressor_params:
            all_metrics[name].append([])

    epochs = []
    is_finished = False
    for file in tqdm(os.listdir(checkpoints_path)):
        full_path = os.path.join(checkpoints_path, file)
        if not (os.path.isfile(full_path) and file.endswith("pth")
                and "model_" in file):
            continue
        checkpoint = torch.load(full_path, map_location=device)
        encoder_prefix = "backbone" if not supervised else "0"
        checkpoint = encoder_cp_from_model_cp(checkpoint, encoder_prefix)
        epoch = int(file.split(".pth")[0].split("_")[-1])
        if epoch == local_args.epochs:
            is_finished = True
        epochs.append(epoch)
        encoder.load_state_dict(checkpoint)
        model = encoder.to(device)
        model.eval()
        latents = []
        ys = []
        transformed_ys = []
        for step, x in enumerate(train_loader):
            x, metadata, _ = x
            left_x = x["surface-lh"].float().to(device, non_blocking=True)
            right_x = x["surface-rh"].float().to(device, non_blocking=True)
            if args.to_predict in metadata.keys():
                y = metadata[args.to_predict]
            else:
                y = x["clinical"][:, index_to_predict]
            transformed_y = label_prepro.transform(
                np.array(y)[:, np.newaxis]).squeeze()
            transformed_ys.append(transformed_y)
            ys.append(y)
            # with torch.cuda.amp.autocast():
            data = (left_x, right_x)
            latents.append(model(data).squeeze().detach().cpu().numpy())
        Y = np.concatenate(transformed_ys)
        real_Y = np.concatenate(ys)
        X = np.concatenate(latents)

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
            transformed_valid_y = label_prepro.transform(
                np.array(y)[:, np.newaxis]).squeeze()
            valid_transformed_ys.append(transformed_valid_y)
            valid_ys.append(y)
            # with torch.cuda.amp.autocast():
            data = (left_x, right_x)
            valid_latents.append(model(data).squeeze().detach().cpu().numpy())

        X_valid = np.concatenate(valid_latents)
        Y_valid = np.concatenate(valid_transformed_ys)
        real_y_valid = np.concatenate(valid_ys)

        for value_idx, value in enumerate(regressor_params):
            all_regressor_params = {param_name: value}
            if args.method == "classification":
                all_regressor_params["max_iter"] = 10000
            local_regressor = regressor(**all_regressor_params)
            local_regressor.fit(X, Y)
            y_hat_valid = local_regressor.predict(X_valid)
            real_preds = out_to_real_pred_func(y_hat_valid)

            for name, metric in evaluation_metrics.items():
                all_metrics[name][value_idx].append(metric(Y_valid, y_hat_valid.squeeze()))
            for name, metric in evaluation_against_real_metric.items():
                all_metrics[name][value_idx].append(metric(real_y_valid, real_preds))
    
    if len(epochs) > 0:
        best_epoch_per_metric = {}
        best_value_per_metric = {}
        best_param_value_per_metric = {}
        for name in all_metrics.keys():
            best_epoch_per_metric[name] = []
            best_value_per_metric[name] = []
            for value_idx, _ in enumerate(regressor_params):
                sorted_indices = np.argsort(all_metrics[name][value_idx])
                best_index = 0 if best_is_low else -1
                best_epoch = epochs[sorted_indices[best_index]]
                best_value = all_metrics[name][value_idx][sorted_indices[best_index]]
                best_epoch_per_metric[name].append(best_epoch)
                best_value_per_metric[name].append(best_value)
            sorted_indices = np.argsort(best_value_per_metric[name])
            best_index = 0 if best_is_low else -1
            best_epoch = best_epoch_per_metric[name][sorted_indices[best_index]]
            best_value = best_value_per_metric[name][sorted_indices[best_index]]
            best_param_value = regressor_params[sorted_indices[best_index]]
            best_epoch_per_metric[name] = best_epoch
            best_value_per_metric[name] = best_value
            best_param_value_per_metric[name] = best_param_value
        print(str(setup_id) + ",".join([f" best {name} : {value}"
              for name, value in best_value_per_metric.items()]))
    all_metrics["epochs"] = epochs
    if is_finished:
        setups = pd.read_table(args.setups_file)
        setups.loc[setups["id"] == setup_id, [
            f"best_epoch_{args.to_predict}",
            f"best_param_{args.to_predict}",
            f"best_value_{args.to_predict}"]] = (
                best_epoch_per_metric[validation_metric],
                best_param_value_per_metric[validation_metric],
                best_value_per_metric[validation_metric])
        setups.to_csv(args.setups_file, index=False, sep="\t")
        with open(path_to_metrics, 'w') as fp:
            json.dump(all_metrics, fp)
