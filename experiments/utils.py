import parse

def params_from_args(params, args):
    supervised = False
    old_format = (
        "deepint_barlow_{}_surf_{}_features_fusion_{}_act_{}_bn_{}_conv_{}"
        "_latent_{}_wd_{}_{}_epochs_lr_{}_bs_{}_ba_{}_ima_{}_gba_{}_cutout_{}"
        "_normalize_{}_standardize_{}")
    new_format = (
        "pretrain_{}_on_{}_surf_order_{}_with_{}_features_fusion_{}_act_{}"
        "_bn_{}_conv_{}_latent_{}_wd_{}_{}_epochs_lr_{}_reduced_{}_bs_{}_ba_{}"
        "_ima_{}_blur_{}_noise_{}_cutout_{}_normalize_{}_standardize_{}_"
        "loss_param_{}_sigma_{}")
    new_new_format = (
        "pretrain_{}_on_{}_surf_order_{}_with_{}_features_fusion_{}_act_{}"
        "_bn_{}_conv_{}_latent_{}_wd_{}_{}_epochs_lr_{}_reduced_{}_bs_{}_ba_{}"
        "_ima_{}_blur_{}_noise_{}_cutout_{}_normalize_{}_standardize_{}_"
        "loss_param_{}_sigma_{}_projector_{}")
    format_from_predict = ("predict_{}_with_{}_on_{}_surf_order_{}_with_{}_"
        "features_fusion_{}_act_{}_bn_{}_conv_{}_latent_{}_wd_{}_{}_epochs_"
        "lr_{}_reduced_{}_bs_{}_normalize_{}_standardize_{}_loss_{}_weighted"
        "_{}_mixup_{} _momentum_{}_fut_{}_pretrained_setup_{}_epoch_{}_dr_{}_"
        "nlp_{}")
    old = True
    new = True
    if params.startswith("predict"):
        print("here")
        print(params)
        args_names = ["to_predict", "method", "data_train", "ico_order",
            "n_features", "fusion_level", "activation", "batch_norm",
            "conv_filters", "latent_dim", "weight_decay", "epochs",
            "learning_rate", "reduce_lr", "batch_size", "normalize",
            "standardize", "loss", "weight_criterion", "mixup", "momentum",
            "freeze_up_to", "pretrained_setup", "pretrained_epoch",
            "dropout_rate", "n_layers_predictor"]
        parsed = parse.parse(format_from_predict, params)
        supervised = True
    else:
        try:
            parsed = parse.parse(old_format, params.split("/")[-2])
        except Exception:
            old = False
            try:
                parsed = parse.parse(new_new_format, params)
                if parsed is None:
                    raise Exception()
            except Exception:
                new = False
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
            if new: 
                args_names.append("projector")
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
