# MixUp brain-cortical augmentations in self-supervised learning

\:+1: If you are using the code please add a star to the repository :+1:

Learning biological markers for a specific brain pathology is often impaired by
 the size of the dataset. With the advent of large open datasets in the general
 population, new learning strategies have emerged. In particular, deep 
representation learning consists of training a model via pretext tasks that can
 be used to solve downstream clinical problems of interest. More recently, 
self-supervised learning provides a rich framework for learning representations
 by contrasting transformed samples. These methods rely on carefully designed 
data manipulation to create semantically similar but syntactically different 
samples. In parallel, domain-specific architectures such as spherical 
convolutional neural networks can learn from cortical brain measures in order 
to reveal original biomarkers. Unfortunately, only a few surface-based 
augmentations exist, and none of them have been applied in a self-supervised 
learning setting. We perform experiments on two open source datasets: Big 
Healthy Brain and Healthy Brain Network. We propose new augmentations for the 
cortical brain: baseline augmentations adapted from classical ones for training 
convolutional neural networks, typically on natural images, and new 
augmentations called MixUp. The results suggest that surface-based self-
supervised learning performs comparably to supervised baselines, but 
generalizes better to different tasks and datasets. In addition, the learned 
representations are improved by the proposed MixUp augmentations.

This is the official repository for our unpublished paper associated code.
If you have any question about the code or the paper, we are happy to help!


## Preliminaries

This code was developed and tested with:
- Python version 3.9.13
- PyTorch version 1.13.0
- CUDA version 11.0
- The conda environment defined in `environment.yml`

First, set up the conda enviroment as follows:

```
conda env create -f environment.yml
conda activate corticalmixup
```

or install the requirements in your own environment. 

In order to be able to run the experiments, you need to have access to HBN or
EUAIMS data. Then, you must provide each script the path to these data setting
the `--dataset` and `--datasetdir` parameters.
The data folder must contains at least 5 files:
- **rois_data.npy**: an array with 2 dimensions, the first corresponding to
  the subjects, the second to the different metric for each ROI.
- **rois_subjects.npy**: the list of subjects with the same ordering as
  in the previous file.
- **roi_names.npy**: the list of feature names for the `roi_data` file, with
  the same ordering as its columns.
- **clinical_data.npy**: an array with 2 dimensions, the first corresponding
  to the subjects, the second to the different score values.
- **clinical_subjects.npy**: the list of subjects with the same ordering as
  in the previous file.
- **clinical_names.npy** the list of feature names for the `clinical_data`
  file, with the same ordering as its columns.
- **metadata.tsv**: a table containing the metadata. It must contain at least
  4 columns: `participant_id` with the id of the subjects, corresponding
  to the `_subjects` files, `sex` with numerically encoded sex, `age` with
  continuous age, and `site` with acquisition site names.


## Experiments

To choose between running the MVAE, MMVAE, and MoPoE-VAE, one needs to
change the script's `--method` variabe to `poe`, `moe`, or `joint_elbo`
respectively. By default, `joint_elbo` is selected.


Perform the proposed Digital Avatars Aanalysis (DAA) on HBN by running
the following commands in a shell:

```
cd experiments
export DATASETDIR=/path/to/my/dataset
export OUTDIR=/path/to/the/output/directory

./experiments train --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR
--latent_dim 20 --input_dims 7,444 --beta 1 --batch_size 256
--likelihood normal --initial_learning_rate 0.002 --n_epochs 550
--learn_output_scale --allow_missing_blocks

export RUN=my_run_id

# /!\ Long run /!\
./experiments daa --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR --run $RUN --n_samples 150 --n_validation 20 --trust_level 0.7
./experiments rse --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR --run $RUN

./experiments daa-plot-most-connected --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR --run $RUN
./experiments daa-plot-score-metric --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR --run $RUN --score SRS_Total --metric thickness
./experiments rsa-plot --dataset hbn --datasetdir $DATASETDIR --outdir $OUTDIR --run $RUN
./experiments hist-plot --datasets hbn,euaims --datasetdirs $DATASETDIR1,$DATASETDIR2 --scores SRS_Total,t1_srs_rawscore --outdir $OUTDIR
```

Citation
========

No paper has yet been published about this work.

