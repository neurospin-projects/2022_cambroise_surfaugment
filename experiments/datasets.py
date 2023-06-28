import os
import torch
import numpy as np
import pandas as pd
from itertools import chain, combinations
from sklearn.model_selection import ShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data.dataloader import (_MultiProcessingDataLoaderIter,
                                         _SingleProcessDataLoaderIter,
                                         _BaseDataLoaderIter)
from .fetchers import *
from .utils import discretizer


class MultimodalDataset(torch.utils.data.Dataset):
    """ Multimodal dataset
    """

    def __init__(self, idx_path, metadata_path=None, indices=None,
                 transform=None, on_the_fly_transform=None,
                 on_the_fly_inter_transform=None, overwrite=False):

        self.idx_per_mod = dict(np.load(idx_path, allow_pickle=True))
        self.modalities = list(self.idx_per_mod)
        self.metadata = (pd.read_table(metadata_path) if metadata_path
                         else None)
        n_samples = [len(self.idx_per_mod[key]) for key in self.modalities]
        if not all([n_samples[i] == n_samples[(i+1) % len(n_samples)]
                    for i in range(len(n_samples))]):
            raise ValueError("All modalities do not have the same number of"
                             "samples.")
        if self.metadata is not None and n_samples[0] != len(self.metadata):
            raise ValueError("The data and metadata do not have the same"
                             "number of samples.")
        if transform is not None and type(transform) is dict:
            if not all([k in self.modalities for k in transform.keys()]):
                raise ValueError("The transform should be either a function,"
                                 "or a dict with modalities as keys and"
                                 "function as values.")
        if (on_the_fly_transform is not None and
           type(on_the_fly_transform) is dict):
            if not all([k in self.modalities for k in on_the_fly_transform]):
                raise ValueError("The transform should be either a function,"
                                 "or a dict with modalities as keys and"
                                 "function as values.")
        self.n_samples = n_samples[0]
        self.indices = indices
        self.modality_subsets = list(chain.from_iterable(
            combinations(self.modalities, n) for n in range(
                1, len(self.modalities)+1)))
        self.idx_per_modality_subset = self.compute_idx_per_modality_subset()

        datasetdir = "/".join(idx_path.split("/")[:-1])
        column_names = {}
        for mod in self.modalities:
            path = os.path.join(datasetdir, "{}_names.npy".format(mod))
            if os.path.exists(path):
                columns = np.load(path, allow_pickle=True)
                columns = [col.replace("&", "_").replace("-", "_")
                           for col in columns]
                column_names[mod] = columns

        data_path = idx_path.replace("idx", "data").replace(".npz", ".npy")
        data_path = data_path.replace("_train", "").replace("_test", "")
        if transform is not None and transform:
            transformed_data = {}
            data_path = data_path.replace(".npy", "_transformed.npy")
            for mod in self.modalities:
                mod_path = data_path.replace("multiblock", mod)
                if overwrite or not os.path.exists(mod_path):
                    orig_mod_path = mod_path.replace("_transformed", "")
                    data = np.load(orig_mod_path, mmap_mode="r")
                    if type(transform) == dict:
                        if mod in transform.keys():
                            if mod in column_names.keys():
                                columns = column_names[mod]
                                mod_metadata = pd.read_table(os.path.join(
                                    datasetdir, "{}_metadata.tsv".format(mod)))
                                df = pd.concat(
                                    [mod_metadata,
                                     pd.DataFrame(data, columns=columns)],
                                    axis=1)
                                transformed_data[mod] = transform[mod](
                                    df)[columns].values
                            else:
                                transformed_data[mod] = transform[mod](data)
                        else:
                            transformed_data[mod] = data
                    else:
                        transformed_data[mod] = transform(data)
                    np.save(mod_path, transformed_data[mod])

        self.data = {}
        for mod in self.modalities:
            mod_path = data_path.replace("multiblock", mod)
            self.data[mod] = np.load(mod_path, mmap_mode="r+")
        self.on_the_fly_transform = on_the_fly_transform
        self.on_the_fly_inter_transform = on_the_fly_inter_transform

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return self.n_samples

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
        idx_per_mod = {
            mod: self.idx_per_mod[mod][idx] for mod in self.modalities}
        ret = {
            mod: self.data[mod][int(idx_per_mod[mod])]
            for mod in self.modalities if idx_per_mod[mod] is not None}
        if self.metadata is not None:
            metadata = self.metadata.iloc[idx].to_dict()
        if self.on_the_fly_transform is not None:
            transform = self.on_the_fly_transform
            for mod in self.modalities:
                if mod in ret.keys():
                    data = ret[mod]
                    data = torch.tensor(data)
                    if type(transform) == dict and mod in transform.keys():
                        ret[mod] = transform[mod](data)
                    elif type(transform) != dict:
                        ret[mod] = transform(data)
        if self.on_the_fly_inter_transform is not None:
            ret = self.on_the_fly_inter_transform(ret)

        return ret, metadata, idx

    def compute_idx_per_modality_subset(self):
        all_idx = list(range(len(self)))
        idx_per_modality_subset = [[] for _ in self.modality_subsets]
        for idx in all_idx:
            modalities = []
            for mod in self.modalities:
                true_idx = idx
                if self.indices is not None:
                    true_idx = self.indices[idx]
                if self.idx_per_mod[mod][true_idx] is not None:
                    modalities.append(mod)
            for sub_idx, subset in enumerate(self.modality_subsets):
                if (all([mod in subset for mod in modalities]) and
                    all([mod in modalities for mod in subset])):
                    idx_per_modality_subset[sub_idx].append(idx)
                    break
        return idx_per_modality_subset

    def get_modality_proportions(self):
        return [len(sub_idx) / len(self) for sub_idx in 
                self.idx_per_modality_subset]


class DataManager(object):
    """ Data manager that builds the datasets
    """
    available_datasets = ["hbn", "openbhb", "privatebhb"]
    fetchers = {
        "hbn": make_hbn_fetchers,
        "openbhb": make_openbhb_fetchers,
        "privatebhb": make_privatebhb_fetchers,
    }
    available_modalities = {
        "hbn": ["clinical", "rois", "surface-lh", "surface-rh"],
        "openbhb": ["surface-lh", "surface-rh"],
        "privatebhb": ["surface-lh", "surface-rh"]
    }
    defaults = {
        "hbn": hbn_defaults,
        "openbhb": openbhb_defaults,
        "privatebhb": privatebhb_defaults,
    }

    def __init__(self, dataset, datasetdir, modalities, transform=None,
                 on_the_fly_transform=None, on_the_fly_inter_transform=None,
                 test_size="defaults", validation=None, val_size=0.2,
                 stratify="defaults", discretize="defaults", seed="defaults",
                 overwrite=False, **fetcher_kwargs):
        if dataset not in self.available_datasets:
            raise ValueError("{} dataset is not available".format(dataset))
        if not all([mod in self.available_modalities[dataset]
                    for mod in modalities]):
            wrong_mods = [mod for mod in modalities
                          if mod not in self.available_modalities[dataset]]
            raise ValueError(
                "{} is not an available modality for {} dataset".format(
                    wrong_mods[0], dataset))

        test_size, stratify, discretize, seed = (
            self.check_parameters_and_load_defaults(
                dataset, test_size, stratify, discretize, seed))

        self.dataset = dataset
        self.modalities = modalities
        self.test_size = test_size
        self.transform = transform
        self.on_the_fly_transform = on_the_fly_transform
        self.on_the_fly_inter_transform = on_the_fly_inter_transform

        if not os.path.isdir(datasetdir):
            os.makedirs(datasetdir)

        fetchers = self.fetchers[dataset](datasetdir)
        self.fetcher = fetchers["multiblock"](
            blocks=modalities, seed=seed, stratify=stratify,
            discretize=discretize, test_size=test_size, overwrite=overwrite,
            **fetcher_kwargs)

        idx_path = self.fetcher.train_input_path
        metadata_path = self.fetcher.train_metadata_path

        train_on_the_fly = on_the_fly_transform
        if (type(train_on_the_fly) is dict and
            "train" in train_on_the_fly.keys()):
            train_on_the_fly = train_on_the_fly["train"]
        if validation is not None:
            assert (type(validation) is int) and (validation > 0)
            idx_per_mod = np.load(idx_path, mmap_mode="r", allow_pickle=True)
            metadata = pd.read_table(metadata_path)
            modalities = list(idx_per_mod)
            full_indices = []
            not_full_indices = []
            for idx in range(len(idx_per_mod[modalities[0]])):
                if True in [indices[idx] is None for indices in
                            idx_per_mod.values()]:
                    not_full_indices.append(idx)
                else:
                    full_indices.append(idx)
            self.train_dataset = {}
            splitter = ShuffleSplit(
                validation, test_size=val_size, random_state=seed)
            y = None
            if stratify is not None:
                assert type(stratify) is list
                splitter = MultilabelStratifiedShuffleSplit(
                    validation, test_size=val_size, random_state=seed)
                y = metadata[stratify].iloc[full_indices].copy()
                for name in stratify:
                    if name in discretize:
                        y[name] = discretizer(y[name].values)
            splitted = splitter.split(full_indices, y)
            for fold, (train_idx, valid_idx) in enumerate(splitted):
                fold_on_the_fly = train_on_the_fly
                if type(fold_on_the_fly) is list:
                    fold_on_the_fly = fold_on_the_fly[fold]
                train_idx = np.array(train_idx.tolist() + not_full_indices)
                self.train_dataset[fold] = {}
                self.train_dataset[fold]["train"] = MultimodalDataset(
                    idx_path, metadata_path,
                    train_idx, transform, fold_on_the_fly,
                    on_the_fly_inter_transform, overwrite)
                self.train_dataset[fold]["valid"] = MultimodalDataset(
                    idx_path, metadata_path,
                    valid_idx, transform, fold_on_the_fly,
                    on_the_fly_inter_transform, overwrite)
                self.train_dataset[fold]["train_idx"] = train_idx
                self.train_dataset[fold]["valid_idx"] = valid_idx
            all_on_the_fly = train_on_the_fly
            if type(all_on_the_fly) is list:
                all_on_the_fly = all_on_the_fly[-1]
            self.train_dataset["all"] = MultimodalDataset(
                idx_path, metadata_path, None,
                transform, all_on_the_fly,
                on_the_fly_inter_transform, overwrite)
        else:
            self.train_dataset = MultimodalDataset(
                idx_path, metadata_path, None, transform,
                train_on_the_fly, on_the_fly_inter_transform,
                overwrite)

        test_on_the_fly = on_the_fly_transform
        if (type(test_on_the_fly) is dict and
            "test" in test_on_the_fly.keys()):
            test_on_the_fly = test_on_the_fly["test"]
        if test_size is None or test_size > 0:
            idx_path = self.fetcher.test_input_path
            metadata_path = self.fetcher.test_metadata_path
            self.test_dataset = MultimodalDataset(
                idx_path, metadata_path, None, transform,
                test_on_the_fly, on_the_fly_inter_transform,
                overwrite)

    def check_parameters_and_load_defaults(self, dataset, test_size, stratify,
                                           discretize, seed):
        if test_size == "defaults":
            test_size = self.defaults[dataset]["multiblock"]["test_size"]
        if not (test_size is None or (test_size >= 0 and test_size < 1)):
            raise ValueError("The test size must be in [0, 1) or None")
        if stratify == "defaults":
            stratify = self.defaults[dataset]["multiblock"]["stratify"]
        if discretize == "defaults":
            discretize = self.defaults[dataset]["multiblock"]["discretize"]
        if seed == "defaults":
            seed = self.defaults[dataset]["multiblock"]["seed"]
        if seed != int(seed):
            raise ValueError("The seed must be an integer")
        return test_size, stratify, discretize, seed


    def create_val_from_test(self, val_size=0.5, seed="defaults",
                             discretize="defaults", stratify="defaults"):

        idx_path = self.fetcher.test_input_path
        metadata_path = self.fetcher.test_metadata_path

        _, stratify, discretize, seed = (
            self.check_parameters_and_load_defaults(
                self.dataset, val_size, stratify, discretize, seed))
        idx_per_mod = np.load(idx_path, mmap_mode="r", allow_pickle=True)
        metadata = pd.read_table(metadata_path)
        modalities = list(idx_per_mod)
        splitter = ShuffleSplit(
            1, test_size=val_size, random_state=seed)
        y = None
        if stratify is not None:
            assert type(stratify) is list
            splitter = MultilabelStratifiedShuffleSplit(
                1, test_size=val_size, random_state=seed)
            y = metadata[stratify].copy()
            for name in stratify:
                if name in discretize:
                    y[name] = discretizer(y[name].values)
        splitted = splitter.split(range(len(idx_per_mod[modalities[0]])), y)
        new_test_dataset = {}
        for (valid_idx, test_idx) in splitted:
            new_test_dataset["valid"] = MultimodalDataset(
                idx_path, metadata_path,
                valid_idx, self.transform, self.on_the_fly_transform,
                self.on_the_fly_inter_transform, False)
            new_test_dataset["test"] = MultimodalDataset(
                idx_path, metadata_path,
                test_idx, self.transform, self.on_the_fly_transform,
                self.on_the_fly_inter_transform, False)
            new_test_dataset["valid_idx"] = valid_idx
            new_test_dataset["test_idx"] = test_idx
            new_test_dataset["all"] = MultimodalDataset(
                idx_path, metadata_path, None, self.transform,
                self.on_the_fly_transform, self.on_the_fly_inter_transform,
                False)
        self.test_dataset = new_test_dataset

    def __getitem__(self, key):
        if key not in ["train", "test"]:
            raise ValueError("The key must be 'train' or 'test'")
        if key == "test" and self.test_size == 0:
            raise ValueError("This dataset does not have test data")
        return self.train_dataset if key == "train" else self.test_dataset



def transform_batch_data(batch_data, transforms):
    data, metadata, index = batch_data
    ret = {mod: data[mod] for mod in data.keys()}
    if transforms is not None:
        for mod in transforms.keys():
            ret[mod] = transforms[mod](data[mod], index)
    return ret, metadata, index


class _MultiProcessingDataLoaderIterTransfo(_MultiProcessingDataLoaderIter):
    def __init__(self, batch_transforms=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_transforms = batch_transforms

    def _next_data(self):
        batch_data = super()._next_data()
        return transform_batch_data(batch_data, self.batch_transforms)


class _SingleProcessDataLoaderIterTransfo(_SingleProcessDataLoaderIter):
    def __init__(self, batch_transforms=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_transforms = batch_transforms

    def _next_data(self):
        batch_data = super()._next_data()
        return transform_batch_data(batch_data, self.batch_transforms)


class DataLoaderWithBatchAugmentation(torch.utils.data.DataLoader):
    def __init__(self, batch_transforms=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_transforms = batch_transforms

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIterTransfo(
                self.batch_transforms, self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIterTransfo(
                self.batch_transforms, self)
