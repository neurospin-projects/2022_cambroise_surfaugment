import numpy as np
import torch
from PIL import ImageFilter
import random
from collections import namedtuple

class Transformer(object):
    """ Class that can be used to register a sequence of transformations.
    Inspired from https://github.com/Duplums/yAwareContrastiveLearning/
    """
    Transform = namedtuple("Transform", ["transform", "probability"])

    def __init__(self, pipelines=None):
        """ Initialize the class.
        """
        assert (pipelines is None or (type(pipelines) is list and
                    all([type(name) is str for name in pipelines])))
        self.transforms = []
        if pipelines is not None:
            self.transforms = {name: [] for name in pipelines}

    def register(self, transform, probability=1, pipeline=None):
        """ Register a new transformation.
        Parameters
        ----------
        transform: callable
            the transformation object.
        probability: float, default 1
            the transform is applied with the specified probability.
        """
        trf = self.Transform(transform=transform, probability=probability)
        assert pipeline is None or pipeline in self.transforms
        if type(self.transforms) is list:
            transforms = [self.transforms]
        elif pipeline is None:
            transforms = list(self.transforms.values())
        else:
            transforms = [self.transforms[pipeline]]
        for transform in transforms:
            transform.append(trf)

    def __call__(self, arr):
        """ Apply the registered transformations.
        """
        transforms = self.transforms
        if type(transforms) is list:
            transforms = dict(one_and_only=transforms)
        all_transformed = []
        for transform in transforms.values():
            transformed = torch.clone(arr)
            for trf in transform:
                if np.random.rand() < trf.probability:
                    transformed = trf.transform(transformed)
            all_transformed.append(transformed)
        if len(transforms) == 1:
            all_transformed = all_transformed[0]
        return all_transformed

    def __str__(self):
        if len(self.transforms) == 0:
            return "(Empty Transformer)"
        if type(transforms) is list:
            s = "Composition of:"
            for trf in self.transforms:
                s += "\n\t- "+trf.__str__()
        else:
            for pipe, transforms in self.transforms.items():
                s = "Pipeline {name} composed of:"
                for trf in self.transforms:
                    s += "\n\t- "+trf.__str__()
                s += "\n"
        return


class RescaleAsImage(object):
    metric_limits = {
        "thickness": [0, 5],
        "curv": [-5, 5], #[-94.7961654663086, 109.49639892578125],
        "sulc": [-16.196352005004883, 18.371854782104492],
    }

    def __init__(self, metrics, feature_range=(0, 255), channel_dim=0):
        self.metrics = metrics
        self.feature_range = feature_range
        self.channel_dim = channel_dim

    def __call__(self, data):
        rescaled_data = data.clone()
        for idx, name in enumerate(self.metrics):
            torch_idx =  torch.LongTensor([[[idx]]])
            X_std = (np.clip(torch.take_along_dim(data, torch_idx, self.channel_dim) - self.metric_limits[name][0], 0, self.metric_limits[name][1] - self.metric_limits[name][0]) / 
                     (self.metric_limits[name][1] - self.metric_limits[name][0]))
            rescaled_values = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
            rescaled_data = rescaled_data.scatter_(self.channel_dim, torch_idx, rescaled_values)
        return rescaled_data

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        return img


class Permute(object):
    def __init__(self, new_order):
        self.order = new_order

    def __call__(self, arr):
        if len(arr.shape) < len(self.order):
            arr = arr.unsqueeze(max(self.order))
        return torch.permute(arr, self.order)

class Reshape(object):
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, arr):
        return arr.view(self.shape)


class Normalize(object):
    def __init__(self, mean=0.0, std=1.0, eps=1e-8):
        self.mean=mean
        self.std=std
        self.eps=eps

    def __call__(self, arr):
        return self.std * ((arr - arr.mean(
            dim=tuple(range(1, len(arr.shape))), keepdim=True)) / (arr.std(
                dim=tuple(range(1, len(arr.shape))), keepdim=True) + self.eps)  + self.mean)


class Cutout(object):
    """Apply a cutout on the images
    cf. Improved Regularization of Convolutional Neural Networks with Cutout, arXiv, 2017
    We assume that the square to be cut is inside the image.
    """
    def __init__(self, patch_size=None, value=0, random_size=False, inplace=False, localization=None, p=0.5):
        self.patch_size = patch_size
        self.value = value
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization
        self.p = p

    def __call__(self, arr):
        if random.random() < self.p:
            img_shape = np.array(arr.shape)
            if type(self.patch_size) == int:
                size = [self.patch_size for _ in range(len(img_shape))]
            else:
                size = np.copy(self.patch_size)
            assert len(size) == len(img_shape), "Incorrect patch dimension."
            indexes = []
            for ndim in range(len(img_shape)):
                if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                    size[ndim] = img_shape[ndim]
                if self.random_size:
                    size[ndim] = np.random.randint(0, size[ndim])
                if self.localization is not None:
                    delta_before = max(self.localization[ndim] - size[ndim]//2, 0)
                else:
                    delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
                indexes.append(slice(int(delta_before), int(delta_before + size[ndim])))
            if self.inplace:
                arr[tuple(indexes)] = self.value
                return arr
            else:
                arr_cut = torch.clone(arr)
                arr_cut[tuple(indexes)] = self.value
                return arr_cut
        return arr


class Bootstrapping(object):
    def __init__(self, p, p_corrupt, across_channels=True,
                 groups=None, normalizer=Normalize()):
        self.p = p
        self.p_corrupt = p_corrupt
        self.across_channels = across_channels
        self.groups = groups
        self.normalizer = normalizer

    def __call__(self, batch_data, batch_idx=None):

        return_single = False
        if type(batch_data) not in [tuple, list]:
            batch_data = [batch_data]
            return_single = True
        
        if type(self.p) not in [list, tuple]:
            self.p = [self.p] * len(batch_data)

        if batch_idx is not None:
            batch_idx = batch_idx.tolist()
        new_data = []
        for idx, x in enumerate(batch_data):
            n_samples, n_channels, img_height, img_width = x.shape
            idx_to_permute = range(n_samples)
            if self.groups is not None:
                idx_to_permute = [list(set(self.groups[sample_idx]).intersection(batch_idx)) for sample_idx in batch_idx]
                idx_to_permute_in_batch = [[batch_idx.index(index) for index in sublist] + [num] for num, sublist in enumerate(idx_to_permute)]
            x_tilde = x
            if np.random.random() < self.p[idx]:
                min_corruption = 0.1
                corruption_level = np.random.random() * (self.p_corrupt - min_corruption) + min_corruption
                p_change_selected_feature = 0.6
                p_select_feature = corruption_level / p_change_selected_feature
                
                mask_features = np.random.binomial(1, p_select_feature, (img_height, img_width))
                selected_feature_idx = np.where(mask_features == 1)
          
                masks = np.zeros_like(x)
                if self.across_channels:
                    restreined_masks = np.random.binomial(1, p_change_selected_feature, (n_samples, img_height, img_width))
                    restreined_masks = np.repeat(restreined_masks[:, np.newaxis], n_channels, 1)
                else:
                    restreined_masks = np.random.binomial(1, p_change_selected_feature, x.shape)

                masks[:, :, selected_feature_idx[0], selected_feature_idx[1]] = restreined_masks[:, :, selected_feature_idx[0], selected_feature_idx[1]]
                
                x_bar = torch.zeros_like(x)
                for indices in zip(*selected_feature_idx):
                    row, column = indices
                    if self.across_channels:
                        if self.groups is None:
                            new_idx = np.random.permutation(idx_to_permute)
                        else:
                            new_idx = []
                            for sample_idx in range(n_samples):
                                rand_idx = int(np.random.random()*len(idx_to_permute[sample_idx]))
                                new_idx.append(idx_to_permute_in_batch[sample_idx][rand_idx])
                            new_idx = np.array(new_idx)
                        x_bar[:, :, row, column] = x[new_idx, :, row, column]
                    else:
                        for channel in range(n_channels):
                            new_idx = np.random.permutation(idx_to_permute)
                            x_bar[:, channel, row, column] = x[new_idx, channel, row, column]

                # Corrupt samples
                x_tilde = x * (1-masks) + x_bar * masks
                
            if self.normalizer is not None:
                x_tilde = self.normalizer(x_tilde)

            new_data.append(x_tilde)
        if return_single:
            new_data = new_data[0]
        return new_data

class PermuteBeetweenModalities(object):
    def __init__(self, p, p_corrupt, modalities, across_channels=True,
                 normalizer=Normalize()):
        self.p = p
        self.p_corrupt = p_corrupt
        self.across_channels = across_channels
        self.modalities = modalities
        self.normalizer = normalizer

        if type(modalities) not in [list, tuple] or len(modalities) != 2:
            raise ValueError("modalities must contain exactly 2 data modalities.")

    def __call__(self, data):

        mod1 = data[self.modalities[0]]
        mod2 = data[self.modalities[1]]

        if type(mod1) != type(mod2):
            raise ValueError("Both modalities must be of the same type.")

        return_single = False
        if type(mod1) not in [tuple, list]:
            mod1 = [mod1]
            mod2 = [mod2]
            return_single = True
        
        if type(self.p) not in [list, tuple]:
            self.p = [self.p] * len(mod1)

        if type(self.p_corrupt) not in [list, tuple]:
            self.p_corrupt = [self.p_corrupt] * len(mod1)

        
        if True in [mod1[i].shape != mod2[i].shape for i in range(len(mod1))]:
            raise ValueError("Both modalities must have the same dimensions.")
        
        new_mod1 = []
        new_mod2 = []
        min_corruption = 0.05
        for idx in range(len(mod1)):
            shape = mod1[idx].shape
            n_channels = shape[0]
            new_data1 = mod1[idx]
            new_data2 = mod2[idx]
            if np.random.random() < self.p[idx]:
                corruption_level = (np.random.random() * 
                (self.p_corrupt[idx] - min_corruption) + min_corruption)
                if self.across_channels:
                    mask = np.random.binomial(
                        1, corruption_level, np.prod(shape[1:]))
                    mask = np.repeat(mask[np.newaxis], n_channels, 0)
                else:
                    mask = np.random.binomial(*
                        1, corruption_level,
                        (n_channels, np.prod(shape[1:])))
                mask = mask.reshape(shape[1:])
                # Corrupt samples
                new_data1 = mod1[idx] * (1 - mask) + mod2[idx] * mask
                new_data2 = mod2[idx] * (1 - mask) + mod1[idx] * mask

            if self.normalizer is not None:
                new_data1 = self.normalizer(new_data1)
                new_data2 = self.normalizer(new_data2)

            new_mod1.append(new_data1)
            new_mod2.append(new_data2)
        
        if return_single:
            new_mod1 = new_mod1[0]
            new_mod2 = new_mod2[0]
        data[self.modalities[0]] = new_mod1
        data[self.modalities[1]] = new_mod2
        return data
