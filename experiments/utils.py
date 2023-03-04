import parse

def params_from_args(params, args):
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
    old = True
    new = True
    try:
        parsed = parse.parse(old_format, params.split("/")[-2])
    except Exception:
        old = False
        try:
            parsed = parse.parse(new_new_format, params)
            print(parsed.fixed)
        except Exception:
            new = False
            print("here")
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
    return args