import parse
import numpy as np
import pandas as pd


def extract_and_order_by(df, column_name, values):
    """
    """
    df = df[df[column_name].isin(values)]
    new_index = pd.Series(pd.Categorical(
        df[column_name], categories=values, ordered=True)
        ).sort_values().index
    df = df.reset_index(drop=True).loc[new_index]

    return df


def discretizer(values, method="auto"):
    """
    """
    bins = np.histogram_bin_edges(values, bins=method)
    new_values = np.digitize(values, bins=bins[1:], right=True)

    return new_values


def params_from_args(params, args):
    supervised = False
    ssl_format = (
        "train_ssl_{}_on_{}_surf_order_{}_with_{}_features_fusion_{}_act_{}"
        "_bn_{}_conv_{}_latent_{}_wd_{}_{}_epochs_lr_{}_reduced_{}_bs_{}_gm_{}"
        "_hm_{}_blur_{}_noise_{}_cutout_{}_normalize_{}_standardize_{}_"
        "loss_param_{}_projector_{}")
    supervised_format = ("predict_{}_with_{}_on_{}_surf_order_{}_with_{}_"
        "features_fusion_{}_act_{}_bn_{}_conv_{}_latent_{}_wd_{}_{}_epochs_"
        "lr_{}_reduced_{}_bs_{}_normalize_{}_standardize_{}_loss_{}_weighted"
        "_{}_momentum_{}_fut_{}_pretrained_setup_{}_epoch_{}_dr_{}_"
        "nlp_{}")
    old = True
    new = True
    if params.startswith("predict"):
        args_names = ["to_predict", "method", "data_train", "ico_order",
            "n_features", "fusion_level", "activation", "batch_norm",
            "conv_filters", "latent_dim", "weight_decay", "epochs",
            "learning_rate", "reduce_lr", "batch_size", "normalize",
            "standardize", "loss", "weight_criterion", "momentum",
            "freeze_up_to", "pretrained_setup", "pretrained_epoch",
            "dropout_rate", "n_layers_predictor"]
        parsed = parse.parse(supervised_format, params)
        supervised = True
    else:
        parsed = parse.parse(ssl_format, params)
        
        args_names = [
            "algo", "data_train", "ico_order", "n_features", "fusion_level",
            "activation", "batch_norm", "conv_filters",
            "latent_dim", "weight_decay", "epochs", "learning_rate",
            "reduce_lr", "batch_size", "groupmixup",
            "hemimixup", "blur", "noise", "cutout",
            "normalize", "standardize", "loss_param", "projector"]
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
    return args, supervised

def encoder_cp_from_model_cp(checkpoint, encoder_prefix="backbone"):
    checkpoint = {".".join(key.split(".")[1:]): value 
                for key, value in checkpoint["model_state_dict"].items() if key.startswith(encoder_prefix)}
    return checkpoint
