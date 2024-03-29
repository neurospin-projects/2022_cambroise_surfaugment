import argparse
import json
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, OrdinalEncoder
import pandas as pd
import torch
from torch import nn, optim
from torchvision import transforms
from surfify.models import SphericalHemiFusionEncoder
from surfify.utils import setup_logging, icosahedron, downsample_data, downsample

from datasets import DataManager
from augmentations import Normalize, Reshape, Transformer
from utils import params_from_args


# Get user parameters
parser = argparse.ArgumentParser(description="Spherical predictor")
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
    "--latent-dim", default=128, type=int, metavar="N",
    help="the number of latent dimension.")
parser.add_argument(
    "--outdir", metavar="DIR", help="output directory path.", required=True)
parser.add_argument(
    "--print-freq", default=1, type=int,
    help="printing frequence (as number of epochs) during training.")
parser.add_argument(
    "--to-predict", default="age",
    help="the name of the variable to predict.")
parser.add_argument(
    "--method", default="regression", choices=("regression", "classification"),
    help="the prediction method.")
parser.add_argument(
    "--loss", default="mse", choices=("mse", "l1"),
    help="the loss function.")
parser.add_argument(
    "--dropout-rate", "-dr", default=0, type=float,
    help="the dropout rate applied before predictor.")
parser.add_argument(
    "--n-layers-predictor", "-nlp", default=1, type=int,
    help="the number of layers in the predictor.")
parser.add_argument(
    "--pretrained-setup", default="None",
    help="the path to the pretrained encoder.")
parser.add_argument(
    "--pretrained-epoch", default=None, type=int,
    help="the pretraining epoch at which to load the model.")
parser.add_argument(
    "--setups-file", default="None",
    help="the path to the file linking the setups to the pretrained encoder's path.")
parser.add_argument(
    "--freeze-up-to", default=0, type=int,
    help="optionnally freezes the backbone network's weights up to some layer."
)
parser.add_argument(
    "--save-freq", default=10, type=int,
    help="saving frequence (as number of epochs) during training.")
parser.add_argument(
    "--momentum", default=0.0, type=float,
    help="optionnally uses SGD with momentum.")
parser.add_argument(
    "--reduce-lr", action="store_true",
    help="optionnally reduces the learning rate during training.")
parser.add_argument(
    "--normalize", action="store_true",
    help="optionnally normalizes input.")
parser.add_argument(
    "--standardize", action="store_true",
    help="optionnally standardize input with statistics computed across"
         "the train dataset.")
parser.add_argument(
    "--weight-criterion", action="store_true",
    help="optionnally weights the criterion according to class unbalance.")
args = parser.parse_args()
args.ngpus_per_node = torch.cuda.device_count()
args.ico_order = 5
modalities = ["surface-lh", "surface-rh"]
metrics = ["thickness", "curv", "sulc"]
args.conv = "DiNe"
n_features = len(metrics)
activation = "ReLU"
args.data = "openbhb"
if args.method == "classification":
    args.loss = "ce"
args.conv_filters = [int(item) for item in args.conv_filters.split("-")]

params = ("predict_{}_with_{}_on_{}_surf_order_{}_with_{}_features_fusion_{}_act_{}"
    "_bn_{}_conv_{}_latent_{}_wd_{}_{}_epochs_lr_{}_reduced_{}_bs_{}"
    "_normalize_{}_standardize_{}_loss_{}_weighted_{}_momentum_{}"
    "_fut_{}_pretrained_setup_{}_epoch_{}_dr_{}_nlp_{}").format(
        args.to_predict, args.method, args.data, args.ico_order, n_features, args.fusion_level,
        activation, args.batch_norm, "-".join([str(s) for s in args.conv_filters]),
        args.latent_dim, args.weight_decay, args.epochs, args.learning_rate,
        args.reduce_lr, args.batch_size, args.normalize, args.standardize,
        args.loss, args.weight_criterion, args.momentum,
        args.freeze_up_to, args.pretrained_setup, args.pretrained_epoch,
        args.dropout_rate, args.n_layers_predictor)

# Prepare process
run_id = int(time.time())
setup_logging(level="info", logfile=None)
checkpoint_dir = os.path.join(args.outdir, "supervised_scnns")
checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints", str(run_id))
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(os.path.join(checkpoint_dir, "params.json"), "w") as file:
    json.dump(vars(args), file)
stats_file = open(os.path.join(checkpoint_dir, "stats.txt"), "a", buffering=1)
print(" ".join(sys.argv))
print(" ".join(sys.argv), file=stats_file)

setups_path = os.path.join(args.outdir, "supervised_scnns", "setups.tsv")
if not os.path.exists(setups_path):
    setups = pd.DataFrame({"id": [], "args": [], "best_epoch": [], "best_param": [], "best_value": []})
    setups = setups.astype({"id": int})
else:
    setups = pd.read_table(setups_path)

def same_params_but_epochs(args_str):
    epochs = args_str.split("_epochs")[0].split("_")[-1]
    return (params == args_str.replace(
        f"{epochs}_epochs", f"{args.epochs}_epochs"))

setups = pd.concat([
    setups,
    pd.DataFrame({
        "id": [run_id],
        "args": [params],
        "best_epoch": [0],
        "best_param": [1],
        "best_value": [1000]})],
    ignore_index=True)
setups.to_csv(os.path.join(args.outdir, "supervised_scnns", "setups.tsv"),
    index=False, sep="\t")
print(run_id)

# Load the input cortical data
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
    epochs, lr, reduce_lr = args.epochs, args.learning_rate, args.reduce_lr
    args = params_from_args(params, args)
    if args.pretrained_epoch is not None:
        epoch = args.pretrained_epoch
        if epoch == -1:
            epoch = args.epochs
    pretrained_path = os.path.join(
        "/".join(args.setups_file.split("/")[:-1]),
        "checkpoints", args.pretrained_setup,
        "model_epoch_{}.pth".format(int(epoch)))
    if int(args.pretrained_setup) < 10000:
        pretrained_path = params
        pretrained_path = pretrained_path.replace(
            "encoder.pth", "model_epoch_{}.pth".format(int(epoch)))

    args.epochs, args.learning_rate, args.reduce_lr = epochs, lr, reduce_lr
    checkpoint = torch.load(pretrained_path)
    checkpoint = encoder_cp_from_barlow_cp(checkpoint)



input_shape = (len(metrics), len(ico_verts))

test_size = "defaults"
stratify = ["sex", "age", "site"]
validation = None

scaling = args.standardize
normalize = args.normalize

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
if validation is None:
    on_the_fly_transform = dict()
    for modality in modalities:
        transformer = Transformer()
        if scaling:
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
            if scaling:
                transformer.register(scalers["train"][fold][modality])
            if normalize:
                transformer.register(Normalize())
            on_the_fly_transform["train"][fold][modality] = transformer
    on_the_fly_transform["train"].append(dict())
    for modality in modalities:
        transformer = Transformer()
        if scaling:
            transformer.register(scalers["train"][-1][modality])
        if normalize:
            transformer.register(Normalize())
        on_the_fly_transform["train"][-1][modality] = transformer
        on_the_fly_transform["test"][modality] = transformer


batch_transforms = None
batch_transforms_valid = None

# downsampler = wrapper_data_downsampler(args.outdir, to_order=args.ico_order)
kwargs = {
    "surface-rh": {"metrics": metrics,
                   "z_score": False},
    "surface-lh": {"metrics": metrics,
                   "z_score": False},
}

all_modalities = modalities.copy()

dataset = DataManager(dataset=args.data, datasetdir=args.datadir,
                      modalities=all_modalities, validation=validation,
                      stratify=stratify, discretize=["age"],
                      transform=transform, 
                      on_the_fly_transform=on_the_fly_transform,
                      overwrite=False, test_size=test_size, **kwargs)
dataset.create_val_from_test(
    val_size=0.5, stratify=stratify, discretize=["age"])

valid_loaders = []
if validation is None:
    train_loaders = [
        torch.utils.data.DataLoader(
            dataset["train"], batch_size=args.batch_size, num_workers=6,
            pin_memory=True, shuffle=True),
        torch.utils.data.DataLoader(
            dataset["train"], batch_size=args.batch_size, num_workers=6,
            pin_memory=True, shuffle=True)]
    valid_loaders.append(torch.utils.data.DataLoader(
        dataset["test"]["valid"], batch_size=args.batch_size, num_workers=6,
        pin_memory=True, shuffle=True))
else:
    train_loaders = []
    for fold in range(validation):
        train_loaders.append(torch.utils.data.DataLoader(
            dataset["train"][fold]["train"], batch_size=args.batch_size,
            num_workers=6, pin_memory=True, shuffle=True))
        valid_loaders.append(torch.utils.data.DataLoader(
            dataset["train"][fold]["valid"], batch_size=args.batch_size,
            num_workers=6, pin_memory=True, shuffle=True))
    train_loaders.append(torch.utils.data.DataLoader(
            dataset["train"]["all"], batch_size=args.batch_size,
            num_workers=6, pin_memory=True, shuffle=True))
test_dataset = dataset["test"]["test"]
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, num_workers=6,
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
if "clinical" in all_modalities:
    clinical_names = np.load(os.path.join(args.datadir, "clinical_names.npy"), allow_pickle=True)
    # print(clinical_names)
for idx, loader in enumerate(train_loaders):
    all_label_data.append([])
    for data in loader:
        data, metadata, _ = data
        if args.to_predict in metadata.keys():
            all_label_data[idx].append(metadata[args.to_predict])
        else:
            index_to_predict = clinical_names.tolist().index(args.to_predict)
            all_label_data[idx].append(data["clinical"][:, index_to_predict])
    all_label_data[idx] = np.concatenate(all_label_data[idx])
label_values = np.unique(np.concatenate(all_label_data))

def corr_metric(y_true, y_pred):
    mat = np.concatenate([y_true[:, np.newaxis], y_pred[:, np.newaxis]], axis=1)
    corr_mat = np.corrcoef(mat, rowvar=False)
    return corr_mat[0, 1]

output_activation = nn.Identity()
hidden_dim = 256
tensor_type = "float"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_to_pred_func = lambda x: x.cpu().detach().numpy()
out_to_real_pred_func = []
root_mean_squared_error = lambda x, y: mean_squared_error(x, y, squared=False)
evaluation_against_real_metric = {"real_rmse": root_mean_squared_error, "real_mae": mean_absolute_error, "r2": r2_score, "correlation": corr_metric}
label_prepro = []
if args.method == "regression":
    output_dim = 1
    for idx in range(len(train_loaders)):
        preprocessor = StandardScaler()
        preprocessor.fit(all_label_data[idx][:, np.newaxis])
        label_prepro.append(preprocessor)
        out_to_real_pred_func.append(
            lambda x: preprocessor.inverse_transform(
                x.cpu().detach().unsqueeze(1)).squeeze())
    # output_activation = DifferentiableRound(label_prepro.scale_)
    criterion = nn.MSELoss()
    if args.loss == "l1":
        criterion = nn.L1Loss()
    evaluation_metrics = {"rmse": root_mean_squared_error, "mae": mean_absolute_error}
    
else:
    output_dim = len(label_values)
    evaluation_metrics = {"accuracy": accuracy_score, "bacc": balanced_accuracy_score,
                          "auc": roc_auc_score}
    tensor_type = "long"
    # tensor_type = "float"
    n_bins = 100
    output_activation = nn.Softmax(dim=1)
    # output_activation = nn.Sigmoid()
    # output_activation = nn.Identity()
    output_dim = 2
    for idx in range(len(train_loaders)):
        label_prepro.append(TransparentProcessor())
        # out_to_real_pred_func.append(
        #     lambda x : x.argmax(1).cpu().detach().numpy())
        out_to_real_pred_func.append(
            lambda x : x.round().cpu().detach().numpy())
        if any([type(value) is np.str_ for value in label_values]):
            preprocessor = OrdinalEncoder()
            preprocessor.fit(all_label_data[idx][:, np.newaxis])
            out_to_real_pred_func[idx] = lambda x : preprocessor.inverse_transform(
                x.argmax(1).cpu().detach().unsqueeze(1).numpy()).squeeze()
        if output_dim > n_bins:
            preprocessor = KBinsDiscretizer(n_bins=n_bins, encode="ordinal")
            preprocessor.fit(all_label_data[idx][:, np.newaxis])
            print(label_prepro[idx].bin_edges_)
            output_dim = n_bins
        label_prepro[idx] = preprocessor
    weight = None
    weight_pos = None
    if args.weight_criterion:
        pos_value = 2 if args.to_predict == "asd" else 1
        numb_pos = (all_label_data[-1] == pos_value).sum()
        numb_neg = len(all_label_data[-1]) - numb_pos
        weight_pos = numb_neg / numb_pos
        print(weight_pos)
        weight = torch.Tensor([1, weight_pos]).to(device)
        weight_pos = torch.Tensor([weight_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_pos)
    criterion = nn.CrossEntropyLoss(weight=weight)
    evaluation_against_real_metric = {}
    out_to_pred_func = lambda x: x.argmax(1).cpu().detach().numpy()
    # out_to_pred_func = lambda x: nn.functional.sigmoid(x).round().cpu().detach().numpy()


n_features = len(metrics)

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


all_metrics = {}
for name in evaluation_metrics.keys():
    all_metrics[name] = [[] for _ in range(args.epochs)]
for name in evaluation_against_real_metric.keys():
    all_metrics[name] = [[] for _ in range(args.epochs)]

for fold, (train_loader, test_loader) in enumerate(
    zip(train_loaders, valid_loaders + [test_loader])):
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
            args.freeze_up_to = len(all_encoder_params) + args.freeze_up_to
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

    predictor_layers = []
    last_dim = args.latent_dim
    next_dim = hidden_dim
    act = getattr(nn, activation)()
    for i in range(args.n_layers_predictor):
        if i == args.n_layers_predictor - 1:
            next_dim = output_dim
            act = output_activation
        predictor_layers += [
            nn.Dropout(args.dropout_rate),
            nn.Linear(last_dim, next_dim),
            act]
        last_dim = next_dim

    predictor = nn.Sequential(*predictor_layers)
    # print(predictor)

    model = nn.Sequential(encoder, predictor)

    model = model.to(device)

        # print(model)
    print("Number of trainable parameters : ",
        sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = optim.Adam(model.parameters(), args.learning_rate,
                        weight_decay=args.weight_decay)
    if args.momentum > 0:
        optimizer = optim.SGD(model.parameters(), args.learning_rate,
            momentum=args.momentum, weight_decay=args.weight_decay)

    if args.reduce_lr:
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(loader) * args.epochs)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,#int(
                #np.ceil(len(loader) / args.batch_size) * args.epochs * 0.3),
            gamma=0.1)
        # scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.97 if epoch % 5 == 0 else 1)

    # linear_model = LogisticRegression()
    start_epoch = 1
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs + 1):
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
                postfix={"loss": 0, "lr": stats["lr"], "average time": 0}, disable=not show_pbar) as pbar:
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
                new_y = label_prepro[fold].transform(np.array(y)[:, np.newaxis])
                new_y = getattr(torch.tensor(new_y), tensor_type)().squeeze()
                new_y = new_y.to(device, non_blocking=True)
                if args.to_predict == "asd":
                    new_y -= 1
                if args.to_predict == "sex":
                    new_y[new_y == -1] = 0

                optimizer.zero_grad()
                # with torch.cuda.amp.autocast():
                X = (left_x, right_x)
                y_hat = model(X).squeeze()
                loss = criterion(y_hat, new_y)

                preds = out_to_pred_func(y_hat)
                real_preds = out_to_real_pred_func[fold](y_hat)

                # loss.backward()
                # optimizer.step()
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

        # Validation
        model.eval()
        with tqdm(total=len(test_loader), desc="Validation epoch : {} / {}".format(epoch, args.epochs),
                postfix={"loss": 0, "average time": 0}, disable=not show_pbar) as pbar:
            for step, x in enumerate(test_loader):
                x, metadata, _ = x
                start_batch_time = time.time()
                left_x = x["surface-lh"].float().to(device, non_blocking=True)
                right_x = x["surface-rh"].float().to(device, non_blocking=True)
                if args.to_predict in metadata.keys():
                    y = metadata[args.to_predict]
                else:
                    y = x["clinical"][:, index_to_predict]
                new_y = label_prepro[fold].transform(np.array(y)[:, np.newaxis])
                new_y = getattr(torch.tensor(new_y), tensor_type)().squeeze()
                new_y = new_y.to(device, non_blocking=True)
                if args.to_predict == "asd":
                    new_y -= 1
                if args.to_predict == "sex":
                    new_y[new_y == -1] = 0

                with torch.no_grad():
                    # with torch.cuda.amp.autocast():
                    X = (left_x, right_x)
                    y_hat = model(X).squeeze()
                    loss = criterion(y_hat, new_y)
                    preds = out_to_pred_func(y_hat)
                    real_preds = out_to_real_pred_func[fold](y_hat)

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
        if args.reduce_lr:
            scheduler.step()

        for name in evaluation_metrics.keys():
            all_metrics[name][epoch - 1].append(stats["validation_" + name])
        for name in evaluation_against_real_metric.keys():
            all_metrics[name][epoch - 1].append(stats["validation_" + name])
        if epoch % args.save_freq == 0 and fold == len(train_loaders) - 1:
            dict_to_save = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if args.reduce_lr else None,
                "loss": stats["loss"],
                "valid_loss": stats["validation_loss"]}
            for name, metric in evaluation_metrics.items():
                dict_to_save[name] = stats[name]
                dict_to_save["validation_" + name] = stats["validation_" + name]
            for name, metric in evaluation_against_real_metric.items():
                dict_to_save[name] = stats[name]
                dict_to_save["validation_" + name] = stats["validation_" + name]
            checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{}.pth")
            torch.save(dict_to_save, checkpoint_path.format(epoch))
        
        # all_data = []
        # transformed_ys = []
        # ys = []
        # for step, x in enumerate(train_loader):
        #     x, metadata, _ = x
        #     left_x = x["surface-lh"].float().cpu().detach()
        #     right_x = x["surface-rh"].float().cpu().detach()
        #     if args.to_predict in metadata.keys():
        #         y = metadata[args.to_predict]
        #     else:
        #         y = x["clinical"][:, index_to_predict]
        #     new_y = label_prepro[fold].transform(np.array(y)[:, np.newaxis])
        #     transformed_ys.append(new_y)
        #     ys.append(y)
        #     data = np.concatenate((left_x, right_x), axis=1).reshape((len(left_x), -1))
        #     all_data.append(data)
        # Y = np.concatenate(transformed_ys)
        # real_y = np.concatenate(ys)
        # X = np.concatenate(all_data)
        # print(X.shape)
        # print(real_y.shape)
        # linear_model.fit(X, real_y)
        
        # test_data = []
        # test_ys = []
        # test_transformed_ys = []
        # for step, x in enumerate(test_loader):
        #     x, metadata, _ = x
        #     left_x = x["surface-lh"].float().cpu().detach()
        #     right_x = x["surface-rh"].float().cpu().detach()
        #     if args.to_predict in metadata.keys():
        #         y = metadata[args.to_predict]
        #     else:
        #         y = x["clinical"][:, index_to_predict]
        #     new_y = label_prepro[fold].transform(np.array(y)[:, np.newaxis])
        #     test_ys.append(y)
        #     test_transformed_ys.append(new_y)
        #     with torch.cuda.amp.autocast():
        #         data = np.concatenate((left_x, right_x), axis=1).reshape((len(left_x), -1))
        #         test_data.append(data)
        # X_test = np.concatenate(test_data)
        # Y_test = np.concatenate(test_transformed_ys)
        # real_y_test = np.concatenate(test_ys)

        # preds = linear_model.predict(X_test)
        # print(roc_auc_score(real_y_test, preds))
        # print(balanced_accuracy_score(real_y_test, preds))

    print(json.dumps(stats), file=stats_file)
    # print(stats)
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
    average_metrics[metric] = np.mean(np.asarray(all_metrics[metric])[:, :-1], axis=1)
    std_metrics[metric] = np.std(np.asarray(all_metrics[metric])[:, :-1], axis=1)


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

final_value_per_metric = {}
final_valid_value_per_metric = {}
final_valid_std_per_metric = {}
for metric in all_metrics.keys():
    final_value_per_metric[metric] = all_metrics[metric][best_epochs_per_metric[metric][0]][-1]
    final_valid_value_per_metric[metric] = average_metrics[metric][best_epochs_per_metric[metric][0]]
    final_valid_std_per_metric[metric] = std_metrics[metric][best_epochs_per_metric[metric][0]]

print(final_value_per_metric)
with open(os.path.join(checkpoint_dir, 'final_values.json'), 'w') as fp:
    json.dump(final_value_per_metric, fp)

with open(os.path.join(checkpoint_dir, 'final_valid_stds.json'), 'w') as fp:
    json.dump(final_valid_std_per_metric, fp)

with open(os.path.join(checkpoint_dir, 'final_valid_values.json'), 'w') as fp:
    json.dump(final_valid_value_per_metric, fp)

# checkpoint = torch.load(checkpoint_file)
# model.load_state_dict(checkpoint["model"])
    