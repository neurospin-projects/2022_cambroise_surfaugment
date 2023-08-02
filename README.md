# MixUp brain-cortical augmentations in self-supervised learning

:+1: If you are using the code please add a star to the repository :+1:

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

This is the official repository for our paper published in MICCAI workshop MLCN
2023 associated code. Some additional ablation results are available in the
file **_CA_MLCN_2023_ablation.pdf** to compare baseline augmentations.
If you have any question about the code or the paper, we are happy to help!


## Preliminaries

This code was developed and tested with:
- Python version 3.10.10
- PyTorch version 1.15.0
- CUDA version 12.0
- The conda environment defined in `environment.yml`

First, set up the conda enviroment as follows:

```
conda env create -f environment.yml
conda activate corticalmixup
```

or install the requirements in your own environment. 

In order to be able to run the experiments, you need to have access to [HBN](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/)
or [BHB](https://ieee-dataport.org/open-access/openbhb-multi-site-brain-mri-dataset-age-prediction-and-debiasing)
data. Then, you must provide each script the path to these data setting the
`--datadir` parameter.
The data folder must contains at least 8 files:
- **surface-lh_data.npy**: an array with 3 dimensions, the first corresponding 
  to the subjects, the second to the mesh vertices and the last for the 
  different metrics.
- **surface-lh_subjects.npy**: the list of subjects with the same ordering as
  in the previous file.
- **surface-rh_data.npy**: an array with 3 dimensions, the first corresponding 
  to the subjects, the second to the mesh vertices and the last for the 
  different metrics.
- **surface-rh_subjects.npy**: the list of subjects with the same ordering as
  in the previous file.
- **metadata.tsv**: a table containing the metadata. It must contain at least
  4 columns: `participant_id` with the id of the subjects, corresponding
  to the `_subjects` files, `sex` with numerically encoded or string labels for sex, `age` with
  continuous age, and `site` with acquisition site names.

For HBN FSIQ prediction, you would require 3 aditional files:
- **clinical_data.npy**: an array with 2 dimensions, the first corresponding
  to the subjects, the second to the different scores.
- **clinical_subjects.npy**: the list of subjects with the same ordering as
  in the previous file.
- **clinical_names.npy** the list of feature names for the `clinical_data`
  file, with the same ordering as its columns.

For BHB, the fetcher uses by default the provided split train / test in the 
[original paper](https://www.sciencedirect.com/science/article/pii/S1053811922007522)
. If you want to use the same split, you must provide two additional files:
- **train_subjects.tsv**: a table containing the id of the train subjects. It 
  must simply contain a column `participant_id` with the id of the subjects.
- **test_subjects.tsv**: a table containing the id of the test subjects. It 
  must simply contain a column `participant_id` with the id of the subjects.
Otherwise, it will split the data as for HBN.
The external test set for BHB is not openly available, but you can still get
results for the predicted brain age mean absolute error by submitting your 
model [here](https://baobablab.github.io/bhb/challenges/age_prediction_with_site_removal).

## Experiments

By running the following commands in a shell, you will train three self 
supervised scnns models with the three proposed combinations of augmentations,
two supervised scnns (one on age and the other on sex) and evaluate them
for age prediction on BHB internal test set and HBN test set:

```
cd experiments
export BHBDATADIR=/path/to/bhb/dataset
export HBNDATADIR=/path/to/hbn/dataset
export OUTDIR=/path/to/the/output/directory

# /!\ Long run /!\ Launch the training of a SimCLR-SCNN with Base
# augmentations 
python train_ssl.py --datadir $BHBDATADIR --outdir $OUTDIR 
--latent-dim 128 --batch-size 1024 --normalize --standardize 
--cutout --blur --noise --learning-rate 2e-3 --epochs 400 
--loss-param 2

# /!\ Long run /!\ Launch the training of a SimCLR-SCNN with Base + HemiMixUp
# augmentations 
python train_ssl.py --datadir $BHBDATADIR --outdir $OUTDIR 
--latent-dim 128 --batch-size 1024 --normalize --standardize 
--cutout --blur --noise --learning-rate 2e-3 --epochs 400 
--loss-param 2 --hemimixup 0.3

# /!\ Long run /!\ Launch the training of a SimCLR-SCNN with Base + GroupMixUp
# augmentations 
python train_ssl.py --datadir $BHBDATADIR --outdir $OUTDIR 
--latent-dim 128 --batch-size 1024 --normalize --standardize 
--cutout --blur --noise --learning-rate 2e-3 --epochs 400 
--loss-param 2 --groupmixup 0.4

# /!\ Long run /!\ Launch the training of a Age-supervised SCNN
python train_supervised.py --datadir $BHBDATADIR --outdir $OUTDIR 
--latent-dim 128 --batch-size 1024 --normalize --standardize 
--learning-rate 5e-4 --epochs 100 --loss l1

# /!\ Long run /!\ Launch the training of a Sex-supervised SCNN
python train_supervised.py --datadir $BHBDATADIR --outdir $OUTDIR 
--latent-dim 128 --batch-size 1024 --normalize --standardize 
--learning-rate 5e-4 --epochs 100 --to-predict sex --method classification

# Compute validation metrics for each saved SimCLR-SCNN version
# for some prediction task (default age with regression)
python compute_validation_metrics.py --datadir $BHBDATADIR --outdir $OUTDIR 
--setups-file ${OUTDIR}/ssl_scnns/setups.tsv

# Compute validation metrics for each saved supervised SCNN version
# for age prediction task with regression
python compute_validation_metrics.py --datadir $BHBDATADIR --outdir $OUTDIR 
--setups-file ${OUTDIR}/supervised_scnns/setups.tsv

# Evaluate each best supervised SCNNs version for each setup for age prediction
# on BHB external test set
python evaluate_representations.py --data openbhb --datadir $BHBDATADIR 
--outdir $OUTDIR  --setups-file ${OUTDIR}/ssl_scnns/setups.tsv

# Evaluate each best supervised SCNNs version for each setup for age prediction
# on BHB external test set
python evaluate_representations.py --data openbhb --datadir $BHBDATADIR 
--outdir $OUTDIR --setups-file ${OUTDIR}/supervised_scnns/setups.tsv

# Evaluate each best supervised SCNNs version for each setup for age prediction
# on HBN test set
python evaluate_representations.py --data hbn --datadir $HBNDATADIR 
--outdir $OUTDIR  --setups-file ${OUTDIR}/ssl_scnns/setups.tsv

# Evaluate each best supervised SCNNs version for each setup for age prediction
# on HBN test set
python evaluate_representations.py --data hbn --datadir $HBNDATADIR 
--outdir $OUTDIR --setups-file ${OUTDIR}/supervised_scnns/setups.tsv


```

Citation
========

C. Ambroise, V. Frouin, B. Dufumier, E. Duchesnay and A. Grigis, *MixUp brain-cortical augmentation for self-supervised
learning*, Machine Learning in Clinical Neuroimaging (MLCN) 2023.

