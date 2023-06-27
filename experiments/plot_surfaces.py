# -*- coding: utf-8 -*-
"""
Spherical augmentations
=======================

Credit: C Ambroise

A simple example on how to use augmentations in the spherical domain.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from surfify.utils import (icosahedron, neighbors, setup_logging,
                           downsample_data, downsample,
                           min_depth_to_get_n_neighbors)
from surfify.plotting import plot_trisurf
from surfify.augmentation import (SphericalRandomRotation, SphericalRandomCut,
                                  SphericalBlur, SphericalNoise)
from neurocombat_sklearn import CombatModel
from nilearn import datasets, plotting
from augmentations import Transformer, Normalize
from torchvision import transforms
import torch


setup_logging(level="info", logfile=None)

parser = argparse.ArgumentParser(description="Plot surfaces")
parser.add_argument(
    "--data", default="openbhb",
    choices=("hbn", "euaims", "openbhb", "privatebhb"),
    help="the input cohort name.")
parser.add_argument(
    "--datadir", metavar="DIR", help="data directory path.", required=True)
parser.add_argument(
    "--outdir", metavar="DIR", help="output directory path.", required=True)
parser.add_argument(
    "--measure", default="thickness",
    help="the measure to plot.")
parser.add_argument(
    "--order", default=4, type=int,
    help="the order of the icosahedron to plot.")
parser.add_argument(
    "--hemisphere", default="lh", type=str, choices=("lh", "rh"),
    help="the hemisphere to plot.")
args = parser.parse_args()

coords, triangles = icosahedron(order=args.order)
neighs = neighbors(coords, triangles, direct_neighbor=True)
print(coords.shape)
print(triangles.shape)

measures = ["thickness", "curv", "sulc"]

order = 7
ico_verts, _ = icosahedron(order)
down_indices = []
for low_order in range(order - 1, args.order - 1, -1):
    low_ico_verts, _ = icosahedron(low_order)
    down_indices.append(downsample(ico_verts, low_ico_verts))
    ico_verts = low_ico_verts
def transform(x):
    downsampled_data = downsample_data(x, 7 - args.order, down_indices)
    return np.swapaxes(downsampled_data, 1, 2)

measure_idx = measures.index(args.measure)
colormap = cm.bwr
#############################################################################
# Random cuts
# -----------
#
# Display random cut outputs with different parameters.

np.random.seed(3)
path_to_surfaces = os.path.join(args.datadir, f"surface-{args.hemisphere}_data.npy")
other_hemi = "lh" if args.hemisphere == "rh" else "rh"
tri_textures = np.load(path_to_surfaces, mmap_mode="r")
metadata = pd.read_table(path_to_surfaces.replace("data.npy", "metadata.tsv"))
other_path_to_surface = path_to_surfaces.replace(args.hemisphere, other_hemi)
other_tri_textures = np.load(other_path_to_surface, mmap_mode="r")
other_metadata = pd.read_table(
    other_path_to_surface.replace("data.npy", "metadata.tsv"))
subj_idx = np.random.randint(len(tri_textures))
subj_id = metadata["participant_id"].iloc[subj_idx].item()
other_subj_idx = other_metadata["participant_id"].tolist().index(subj_id)


fig, ax = plt.subplots(1, 1, subplot_kw={
    "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
plot_trisurf(coords, triangles, np.zeros(len(coords)),
             colorbar=False, alpha=1, linewidths=0.3, edgecolors="grey",
             color_map=colormap, vmin=-1, vmax=1, fig=fig, ax=ax)
fig.tight_layout()


plot_augs = False
plot_all_surfs = True
plot_groups = False
if plot_all_surfs:
    fsaverage = datasets.fetch_surf_fsaverage()
    tri_texture = transform(tri_textures[subj_idx][None])[0]
    tri_texture = ((tri_texture - tri_texture.mean(axis=1, keepdims=True)) /
                    tri_texture.std(axis=1, keepdims=True))
    other_tri_texture = transform(other_tri_textures[other_subj_idx][None])[0]
    other_tri_texture = ((other_tri_texture - other_tri_texture.mean(axis=1, keepdims=True)) /
                    other_tri_texture.std(axis=1, keepdims=True))

# fig, ax = plt.subplots(3, 2, subplot_kw={
#         "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
# for idx in range(len(measures)):
#     plotting.plot_surf(fsaverage['pial_left'], tri_texture[idx],
#                         hemi='left', view='lateral',
#                         cmap="bwr", axes=ax[idx, 0])
#     plotting.plot_surf(fsaverage['pial_left'], other_tri_texture[idx],
#                         hemi='left', view='lateral',
#                         cmap="bwr", axes=ax[idx, 1])
# plt.subplots_adjust(wpace=0,hspace=0)
# fig.tight_layout()

    fig, ax = plt.subplots(3, 2, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    for idx in range(len(measures)):
        # vmax = max(np.absolute(tri_texture[idx]).max(), np.absolute(other_tri_texture[idx]).max())
        # vmin = -vmax
        plotting.plot_surf(fsaverage['sphere_left'], tri_texture[idx],
                            hemi='left', view='lateral',
                            cmap="bwr", axes=ax[idx, 0])#,
                            # vmin=vmin, vmax=vmax)
        plotting.plot_surf(fsaverage['sphere_left'], other_tri_texture[idx],
                            hemi='left', view='lateral',
                            cmap="bwr", axes=ax[idx, 1])#,
                            # vmin=vmin, vmax=vmax)
    fig.tight_layout()
    plt.subplots_adjust(wspace=-0.4,hspace=-0.22)


    fig, ax = plt.subplots(3, 2, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    for idx in range(len(measures)):
        # vmax = max(np.absolute(tri_texture[idx]).max(), np.absolute(other_tri_texture[idx]).max())
        # vmin = -vmax
        plotting.plot_surf(fsaverage['infl_left'], tri_texture[idx],
                            hemi='left', view='lateral',
                            cmap="bwr", axes=ax[idx, 0])#,
                            # vmin=vmin, vmax=vmax)
        plotting.plot_surf(fsaverage['infl_left'], other_tri_texture[idx],
                            hemi='left', view='lateral',
                            cmap="bwr", axes=ax[idx, 1])#,
                            # vmin=vmin, vmax=vmax)
    fig.tight_layout()
    plt.subplots_adjust(wspace=-0.4,hspace=-0.22)

    np.random.seed(42)
    transformer = Transformer()
    # transformer.register(transforms.ToTensor())
    # transformer.register(Normalize())
    trf = SphericalBlur(
        coords, triangles, None,
        sigma=(1, 2),
        cachedir=os.path.join(args.outdir, "cached_ico_infos"))
    transformer.register(trf, probability=1)
        
    trf = SphericalNoise(sigma=(0.1, 2))
    transformer.register(trf, probability=0.5)

    patch_size = min_depth_to_get_n_neighbors(np.ceil(len(coords) / 4))
    trf = SphericalRandomCut(
        coords, triangles, None,
        patch_size=patch_size,
        cachedir=os.path.join(args.outdir, "cached_ico_infos"))
    transformer.register(trf, probability=0.5)
    transformer.register(lambda x: x.cpu().detach().numpy())

    augmented_texture = transformer(torch.tensor(tri_texture))
    augmented_other_texture = transformer(torch.tensor(other_tri_texture))

    fig, ax = plt.subplots(3, 2, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    for idx in range(len(measures)):
        # vmax = max(np.absolute(tri_texture[idx]).max(), np.absolute(other_tri_texture[idx]).max())
        # vmin = -vmax
        plotting.plot_surf(fsaverage['sphere_left'], augmented_texture[idx],
                            hemi='left', view='lateral',
                            cmap="bwr", axes=ax[idx, 0])#,
                            #vmin=vmin, vmax=vmax)
        plotting.plot_surf(fsaverage['sphere_left'], augmented_other_texture[idx],
                            hemi='left', view='lateral',
                            cmap="bwr", axes=ax[idx, 1])#,
                            #vmin=vmin, vmax=vmax)
    fig.tight_layout()
    plt.subplots_adjust(wspace=-0.4,hspace=-0.22)

if plot_augs:
    tri_texture = transform(tri_textures[subj_idx][None])[0]
    tri_texture = ((tri_texture - tri_texture.mean(axis=1, keepdims=True)) /
                tri_texture.std(axis=1, keepdims=True))
    other_tri_texture = transform(other_tri_textures[other_subj_idx][None])[0]
    other_tri_texture = ((other_tri_texture - other_tri_texture.mean(
        axis=1, keepdims=True)) / other_tri_texture.std(axis=1, keepdims=True))

    vmin = tri_texture[measure_idx].min()
    vmax = tri_texture[measure_idx].max()

    augmentations = []
    print("initializing random cut augmentation...")
    patch_size = min_depth_to_get_n_neighbors(np.ceil(len(coords) / 4))
    aug = SphericalRandomCut(
        coords, triangles, neighs=None, patch_size=patch_size,
        n_patches=1, random_size=True,
        cachedir=os.path.join(args.outdir, "cached_ico_infos"))

    fig, ax = plt.subplots(1, 2, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    plot_trisurf(coords, triangles, tri_texture[measure_idx], fig=fig, ax=ax[0],
                alpha=1, colorbar=False, edgecolors="white",
                linewidths=0.1, vmin=vmin, vmax=vmax, color_map=colormap)#, is_label=True)
    augmented_texture = aug(tri_texture.copy())
    plot_trisurf(coords, triangles, augmented_texture[measure_idx],
                ax=ax[1], fig=fig,
                alpha=1, colorbar=False, edgecolors="white",
                linewidths=0.1, vmin=vmin, vmax=vmax, color_map=colormap)#, is_label=True)
    fig.tight_layout()

    #############################################################################
    # Spherical blur
    # -----------
    #
    # Display blured textures.

    # tri_texture = np.array([[2, 2], [0, 0]]*int(len(coords) / 2)).T

    print("initializing blur augmentation...")
    aug = SphericalBlur(coords, triangles, sigma=2, fixed_sigma=True,
                        cachedir=os.path.join(args.outdir, "cached_ico_infos"))

    fig, ax = plt.subplots(1, 2, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    plot_trisurf(coords, triangles, tri_texture[measure_idx], fig=fig,
                ax=ax[0], alpha=1, colorbar=False, edgecolors="black",
                linewidths=0, vmin=vmin, vmax=vmax, color_map=colormap)

    augmented_texture = aug(tri_texture.copy())
    plot_trisurf(coords, triangles, augmented_texture[measure_idx],
                    ax=ax[1], fig=fig, colorbar=False, 
                    edgecolors="black", linewidths=0,
                    vmin=vmin, vmax=vmax, color_map=colormap)
    fig.tight_layout()

    #############################################################################
    # Spherical nosie
    # -----------
    #
    # Display blured textures.

    print("initializing noise augmentation...")
    aug = SphericalNoise(sigma=1, fixed_sigma=True)

    fig, ax = plt.subplots(1, 2, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    plot_trisurf(coords, triangles, tri_texture[measure_idx], fig=fig, ax=ax[0],
                alpha=1, colorbar=False, edgecolors="black", linewidths=0,
                vmin=vmin, vmax=vmax, color_map=colormap)
    augmented_texture = aug(tri_texture.copy())
    plot_trisurf(coords, triangles, augmented_texture[measure_idx],
                ax=ax[1], fig=fig,
                colorbar=False, edgecolors="black", linewidths=0,
                vmin=vmin, vmax=vmax, color_map=colormap)
    fig.tight_layout()


    fig, ax = plt.subplots(1, 2, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    plot_trisurf(coords, triangles, tri_texture[measure_idx], fig=fig, ax=ax[0],
                alpha=1, colorbar=False, edgecolors="white",
                linewidths=0, vmin=vmin, vmax=vmax, color_map=colormap)#, is_label=True)
    plot_trisurf(coords, triangles, other_tri_texture[measure_idx], fig=fig, ax=ax[1],
                alpha=1, colorbar=False, edgecolors="white",
                linewidths=0, vmin=vmin, vmax=vmax, color_map=colormap)#, is_label=True)
    fig.tight_layout()

if plot_groups:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    transformed_path = path_to_surfaces.replace(".npy", "_transformed.npy")
    transformed_textures = np.load(transformed_path, mmap_mode="r")
    transformed_textures = transformed_textures.reshape((len(transformed_textures), -1))
    scaler = StandardScaler()
    transformed_textures = scaler.fit_transform(transformed_textures)

    residualizer = CombatModel()
    zero_mask = (transformed_textures == 0).sum(0) == len(transformed_textures)
    residualized_textures = transformed_textures.copy()
    residualized_textures[:, ~zero_mask] = residualizer.fit_transform(
        residualized_textures[:, ~zero_mask], metadata[["site"]].values,
        metadata[["sex"]].values, metadata[["age"]].values)
    reductor = PCA(20)
    reducted_residualized = reductor.fit_transform(residualized_textures)
    # reductor = PCA(20)
    # reducted = reductor.fit_transform(transformed_textures)

    # knn = NearestNeighbors()
    # knn.fit(reducted)
    knn_residualized = NearestNeighbors()
    knn_residualized.fit(reducted_residualized)
    # _, neigh_idx = knn.kneighbors(reducted)
    _, neigh_idx = knn_residualized.kneighbors(reducted_residualized)

    # np.random.seed(2)
    subj_idx = np.random.choice(list(range(len(reducted_residualized))), size=301, replace=False)
    ages = metadata["age"].iloc[subj_idx[:-1]]
    vmax = ages.max()
    vmin = 0
    # sites = metadata["site"].iloc[subj_idx[:-1]]
    scatter = ax.scatter(reducted_residualized[subj_idx[:-1], 0],
                        reducted_residualized[subj_idx[:-1], 1],
                        s=200, c=ages, alpha=0.9,
                        vmin=vmin, vmax=vmax)
    ax.scatter(reducted_residualized[subj_idx[-1], 0],
            reducted_residualized[subj_idx[-1], 1],
            s=600, c=metadata["age"].iloc[subj_idx[-1]],
            marker="*", edgecolors="red", vmin=vmin, vmax=vmax,
            alpha=1)
    circle = plt.Circle(reducted_residualized[subj_idx[-1]], 10, color="red",
                        fill=False, linewidth=3)
    ax.add_artist(circle)
    plt.rcParams['legend.title_fontsize'] = 36
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                     loc="lower left", title="Sites",
    #                     prop={'size': 15})
    # ax.add_artist(legend1)
    handles, labels = scatter.legend_elements()#prop="sizes", alpha=0.6, func=lambda x: x/2)
    legend2 = ax.legend(handles, labels, loc="lower left", title="Age",
                        prop={'size': 28}, markerscale=6)
    ax.axis("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    fig, ax = plt.subplots(2, 3, subplot_kw={
            "projection": "3d", "aspect": "auto"}, figsize=(10, 10))
    neighs_indices = neigh_idx[subj_idx[-1]]

    tri_texture = transform(tri_textures[subj_idx[-1]][None])[0]
    tri_texture = ((tri_texture - tri_texture.mean(axis=1, keepdims=True)) /
                tri_texture.std(axis=1, keepdims=True))
    plot_trisurf(coords, triangles, tri_texture[measure_idx], fig=fig, ax=ax[0, 0],
                alpha=1, colorbar=False, edgecolors="white",
                linewidths=0, color_map=colormap)
    for idx, neigh_idx in enumerate(neighs_indices):
        neigh_texture = transform(tri_textures[neigh_idx][None])[0]
        neigh_texture = ((neigh_texture - neigh_texture.mean(axis=1, keepdims=True)) /
                    neigh_texture.std(axis=1, keepdims=True))
        plot_trisurf(coords, triangles, neigh_texture[measure_idx],
                    ax=ax[(idx + 1) // 3, (idx + 1) % 3], fig=fig,
                    alpha=1, colorbar=False, edgecolors="white",
                    linewidths=0, color_map=colormap)
    fig.tight_layout()
plt.show()
