import argparse
import os
import sys
import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error,
                             mean_absolute_error, balanced_accuracy_score,
                             roc_auc_score)
from sklearn.preprocessing import (KBinsDiscretizer, StandardScaler, 
                                   OrdinalEncoder)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import PCA
import pandas as pd
import torch
from torch import nn, optim
from torchvision import transforms
from surfify.utils import (setup_logging, icosahedron, downsample_data,
                           downsample)

from multimodaldatasets.datasets import DataManager
from augmentations import Normalize, Reshape, Transformer


# Get user parameters
parser = argparse.ArgumentParser(description="Spherical predictor")
parser.add_argument(
    "--data", default="hcp", choices=("hbn", "euaims", "hcp", "openbhb", "privatebhb"),
    help="the input cohort name.")
parser.add_argument(
    "--datadir", metavar="DIR", help="data directory path.", required=True)
parser.add_argument(
    "--to-predict", default="age",
    help="the name of the variable to predict.")
parser.add_argument(
    "--method", default="regression", choices=("regression", "classification"),
    help="the prediction method.")

args = parser.parse_args()
args.ico_order = 5

# Prepare process
setup_logging(level="info", logfile=None)

# Load the input cortical data
modalities = ["surface-lh", "surface-rh"]
all_modalities = modalities.copy()
metrics = ["thickness", "curv", "sulc"]
n_features = len(metrics)
transform = None
batch_size = 64


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

if args.data in ["hbn", "euaims"]:
    kwargs["surface-lh"]["symetrized"] = True
    kwargs["surface-rh"]["symetrized"] = True
    if args.to_predict not in ["sex", "age", "site", "asd"]:
        all_modalities.append("clinical")
        kwargs["clinical"] = dict(cols=[args.to_predict])

test_size = "defaults"
stratify = ["sex", "age", "site"]
validation = None
if args.data not in ["openbhb", "privatebhb"]:
    test_size = 0.2
    validation = 5
    if args.to_predict == "asd":
        stratify.append("asd")

datadir = args.datadir
data = args.data
if args.data == "privatebhb":
    datadir = datadir.replace(args.data, "openbhb")
    data = "openbhb"
dataset = DataManager(
    dataset=data, datasetdir=datadir, modalities=all_modalities,
    stratify=stratify, discretize=["age"], transform=transform,
    overwrite=False, test_size=test_size, validation=validation,
    **kwargs)

params_for_validation = {
    "regression": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "classification": {"C": [0.01, 0.1, 1, 10, 100]}
}
validation_metric = "mae"
best_is_low = True

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
            if (not os.path.exists(path_to_scaler) and
                not (args.data == "privatebhb")):
                X[modality] += data[modality].view(
                    (len(data[modality]), -1)).tolist()
    for modality in modalities:
        suffix = f"_fold_{fold_idx}"
        if fold_idx == len(loaders) - 1:
            suffix = ""
        path_to_scaler = os.path.join(
            args.datadir, f"{modality}_scaler{suffix}.save")
        if (not os.path.exists(path_to_scaler) and
            not (args.data == "privatebhb")):
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
    evaluation_metrics = {#"accuracy": accuracy_score,
                          "bacc": balanced_accuracy_score,
                          "auc": roc_auc_score}
    tensor_type = "long"
    n_bins = 3
    evaluation_against_real_metric = {}
    validation_metric = "bacc"
    label_prepro = TransparentProcessor()
    # out_to_real_pred_func = lambda x : x.argmax(1).cpu().detach().numpy()
    if any([type(value) is np.str_ for value in label_values]):
        label_prepro = OrdinalEncoder()
        label_prepro.fit(all_label_data[:, np.newaxis])
        out_to_real_pred_func = lambda x : label_prepro.inverse_transform(x[:, np.newaxis]).squeeze()
        #     x.argmax(1).cpu().detach().unsqueeze(1).numpy()).squeeze()
    elif output_dim > n_bins:
        label_prepro = KBinsDiscretizer(n_bins=n_bins, encode="ordinal")
        label_prepro.fit(all_label_data[:, np.newaxis])
        print(label_prepro.bin_edges_)
        evaluation_metrics = {"bacc": balanced_accuracy_score}
        evaluation_against_real_metric = {
            "mae": mean_absolute_error,
            "r2": r2_score, "correlation": corr_metric}
        # output_dim = n_bins
    
    best_is_low = False
    # out_to_pred_func = lambda x: x.argmax(1).cpu().detach().numpy()
    regressor = LogisticRegression

    
    
on_the_fly_transform = None

# Creating scalers for train, test and if there are valid folds
scalers_for_rep = {mod: None for mod in modalities}
for modality in modalities:
    datadir = args.datadir.replace(
        args.data, "openbhb").replace(args.datadir.split("/")[-2], "ico5")
    path_to_scaler = os.path.join(
        datadir, f"{modality}_scaler.save")
    scaler = joblib.load(path_to_scaler)
    scalers_for_rep[modality] =  transforms.Compose([
        Reshape((1, -1)),
        scaler.transform,
        transforms.ToTensor(),
        torch.squeeze,
        Reshape(input_shape),
    ])
if validation is None:
    scalers = {mod: None for mod in modalities}
    for modality in modalities:
        datadir = args.datadir
        if args.data == "privatebhb":
            datadir = datadir.replace(args.data, "openbhb")
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
elif validation is not None:
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
normalize = True
on_the_fly_transform_for_rep = dict()
for modality in modalities:
    transformer = Transformer()
    transformer.register(scalers_for_rep[modality])
    if normalize:
        transformer.register(Normalize())
    on_the_fly_transform_for_rep[modality] = transformer
if validation is None:
    on_the_fly_transform = dict()
    for modality in modalities:
        transformer = Transformer()
        transformer.register(scalers[modality])
        if normalize:
            transformer.register(Normalize())
        on_the_fly_transform[modality] = transformer
else:
    on_the_fly_transform = dict(train=[], test=dict())
    for fold in range(validation):
        on_the_fly_transform["train"].append(dict())
        for modality in modalities:
            transformer = Transformer()
            transformer.register(scalers["train"][fold][modality])
            if normalize:
                transformer.register(Normalize())
            on_the_fly_transform["train"][fold][modality] = transformer
    on_the_fly_transform["train"].append(dict())
    for modality in modalities:
        transformer = Transformer()
        transformer.register(scalers["train"][-1][modality])
        if normalize:
            transformer.register(Normalize())
        on_the_fly_transform["train"][-1][modality] = transformer
        on_the_fly_transform["test"][modality] = transformer

train_datadir = args.datadir.replace(
        args.data, "openbhb").replace(args.datadir.split("/")[-2], "ico5")
train_rep_dataset = dataset = DataManager(
    dataset="openbhb", datasetdir=train_datadir,
    modalities=modalities, stratify=stratify,
    discretize=["age"], transform=transform,
    on_the_fly_transform=on_the_fly_transform_for_rep,
    overwrite=False, test_size="defaults")

if args.data == "privatebhb":
    dataset = DataManager(
        dataset="openbhb", datasetdir=train_datadir,
        modalities=modalities, stratify=stratify,
        discretize=["age"], transform=transform,
        on_the_fly_transform=on_the_fly_transform,
        overwrite=False, test_size="defaults", **kwargs)

    test_dataset = DataManager(
        dataset=args.data, datasetdir=args.datadir, modalities=all_modalities,
        transform=transform, on_the_fly_transform=on_the_fly_transform,
        overwrite=False, test_size=0, **kwargs)
else:
    dataset = DataManager(
        dataset=args.data, datasetdir=args.datadir, modalities=all_modalities,
        stratify=stratify, discretize=["age"], validation=validation,
        transform=transform, on_the_fly_transform=on_the_fly_transform,
        overwrite=False, test_size=test_size, **kwargs)
if args.data in ["openbhb", "privatebhb"]:
    dataset.create_val_from_test(
        val_size=0.5, stratify=["sex", "age", "site"], discretize=["age"])
if args.data == "privatebhb":
    dataset.test_dataset["test"] = test_dataset["train"]


train_rep_loader = torch.utils.data.DataLoader(
    train_rep_dataset["train"], batch_size=batch_size, num_workers=6,
    pin_memory=True, shuffle=True)
valid_loaders = []
if validation is None:
    train_loaders = [torch.utils.data.DataLoader(
        dataset["train"], batch_size=batch_size, num_workers=6,
        pin_memory=True, shuffle=True),
                     torch.utils.data.DataLoader(
        dataset["train"], batch_size=batch_size, num_workers=6,
        pin_memory=True, shuffle=True)]
    valid_loaders = [torch.utils.data.DataLoader(
        dataset["test"]["valid"], batch_size=batch_size, num_workers=6,
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
            dataset["train"]["all"], batch_size=batch_size,
            num_workers=6, pin_memory=True, shuffle=True))
test_dataset = dataset["test"]
if args.data in ["openbhb", "privatebhb"]:
    test_dataset = test_dataset["test"]
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, num_workers=6,
    pin_memory=True, shuffle=True)

n_runs = 10
final_perfs = {name: [] for name in (
        list(evaluation_metrics) + list(evaluation_against_real_metric))}
for run in range(n_runs):
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
    reductor = PCA(128)

    X = []
    for step, x in enumerate(train_rep_loader):
        x, metadata, _ = x
        left_x = x["surface-lh"].float().detach().cpu().numpy()
        right_x = x["surface-rh"].float().detach().cpu().numpy()
        X.append(np.concatenate((left_x, right_x), axis=1).reshape(
            (len(left_x), -1)))

    X = np.concatenate(X)
    X = reductor.fit(X)

    for fold, (train_loader, test_loader) in enumerate(
        zip(train_loaders, valid_loaders + [test_loader])):
        X = []
        transformed_ys = []
        ys = []
        for step, x in enumerate(train_loader):
            x, metadata, _ = x
            left_x = x["surface-lh"].float().detach().cpu().numpy()
            right_x = x["surface-rh"].float().detach().cpu().numpy()
            if args.to_predict in metadata.keys():
                y = metadata[args.to_predict]
            else:
                y = x["clinical"][:, index_to_predict]
            new_y = label_prepro.transform(np.array(y)[:, np.newaxis]).squeeze()
            transformed_ys.append(new_y)
            ys.append(y)
            X.append(np.concatenate((left_x, right_x), axis=1).reshape(
                (len(left_x), -1)))

        Y = np.concatenate(transformed_ys)
        real_y = np.concatenate(ys)
        X = np.concatenate(X)

        X = reductor.transform(X)
        
        X_test = []
        test_ys = []
        test_transformed_ys = []
        for step, x in enumerate(test_loader):
            x, metadata, _ = x
            left_x = x["surface-lh"].float().detach().cpu().numpy()
            right_x = x["surface-rh"].float().detach().cpu().numpy()
            if args.to_predict in metadata.keys():
                y = metadata[args.to_predict]
            else:
                y = x["clinical"][:, index_to_predict]
            new_y = label_prepro.transform(np.array(y)[:, np.newaxis]).squeeze()
            test_ys.append(y)
            test_transformed_ys.append(new_y)
            X_test.append(np.concatenate((left_x, right_x), axis=1).reshape(
                (len(left_x), -1)))
        X_test = np.concatenate(X_test)
        Y_test = np.concatenate(test_transformed_ys)
        real_y_test = np.concatenate(test_ys)

        X_test = reductor.transform(X_test)

        if fold < len(train_loaders) - 1:
            params = params_for_validation[args.method]
            param_name = list(params)[0]
            for value_idx, value in enumerate(params[param_name]):
                local_params = {param_name: value}
                if args.method == "classification":
                    local_params["max_iter"] = 10000
                local_regressor = regressor(**local_params)
                local_regressor.fit(X, Y)
                y_hat = local_regressor.predict(X_test)

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
            y_hat = local_regressor.predict(X_test)

            preds = out_to_pred_func(y_hat)
            real_preds = out_to_real_pred_func(y_hat)
            
            final_value_per_metric = {}
            final_std_per_metric = {}
            for name, metric in evaluation_metrics.items():
                final_value_per_metric[name] = metric(Y_test, preds)
            for name, metric in evaluation_against_real_metric.items():
                final_value_per_metric[name] = metric(real_y_test, real_preds)
    for metric in final_value_per_metric.keys():
        final_value = final_value_per_metric[metric]
        final_perfs[metric].append(final_value)
for metric in final_perfs.keys():
    final_value = np.mean(final_perfs[metric])
    final_std = np.std(final_perfs[metric])
    print(f"Test {metric} {final_value} +- {final_std}")
