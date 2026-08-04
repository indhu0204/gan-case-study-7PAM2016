# GAN Case Study – 7PAM2016

This repository contains coursework for a generative modelling case study using Generative Adversarial Networks (GANs) across synthetic data, medical images, cybersecurity tabular data, and sketch generation.

The project is split into two broad parts:

- **Part 1:** building and understanding GANs from scratch on simple synthetic 1D/2D data.
- **Part 2:** applying GANs to three contrasting real datasets: OCTMNIST, CICIDS 2017, and Quick, Draw! birthday cake sketches.

## Project overview

The aim of this project is to explore how GAN behaviour changes across different data modalities and modelling setups.

The repository includes:

- PyTorch GANs for low-dimensional synthetic data.
- A DCGAN for retinal OCT image generation using OCTMNIST.
- A fully connected GAN for CICIDS 2017 tabular intrusion-detection features.
- A TensorFlow DCGAN for Quick, Draw! birthday cake sketch generation.
- Figures, notebooks, and supporting source code used in the final report.

## Repository structure

```text
gan-case-study-7PAM2016/
├── data/
│   ├── cicids/
│   │   └── Wednesday-workingHours.pcap_ISCX.csv
│   ├── quickdraw/
│   │   └── birthday_cake.npy
│   ├── raw/
│   │   └── octmnist.npz
│   └── processed/
├── figures/
│   ├── part1/
│   ├── part2_1_oct/
│   ├── part2_2_cicids/
│   └── part2_3_quickdraw/
├── notebooks/
│   ├── part1_toy_2d_gans.ipynb
│   ├── part2_1_octmnist_dcgan.ipynb
│   ├── part2_2_cicids_gan.ipynb
│   └── part2_3_quickdraw_dcgan.ipynb
├── src/
│   ├── toy_gan_models.py
│   ├── image_dcgan_models.py
│   ├── tabular_gan_models.py
│   └── training_loops.py
│   
├── report/(24096112)_Generative modelling case study.pdf
├── requirements.txt
├── README.md
└── LICENSE
```

### Folder summary

- `data/` – local datasets used for experiments.
- `figures/` – exported plots and image grids used in the report.
- `notebooks/` – notebook workflows for each assignment part and case study.
- `src/` – reusable model definitions, training utilities, and lightweight evaluation code.
- `report/` – report material and final write-up assets.

## Assignment coverage

### Part 1 – Building and understanding GANs from scratch

This section focuses on simple synthetic data to build intuition before moving to real datasets.

Tasks covered:

- Reproducing a sine-wave GAN from the tutorial.
- Modelling an additional low-dimensional distribution.
- Modifying architecture choices such as activation functions and network depth.
- Comparing real and generated samples visually.

Relevant notebook:

- `notebooks/part1_toy_2d_gans.ipynb`

Example outputs:

- Sine-wave distribution and generated samples.
- Mixture-of-Gaussians distribution and generated samples.
- Training-loss plots for original and modified GAN variants.

### Part 2 – Real-data GAN case studies

#### 1. OCTMNIST retinal DCGAN

A convolutional GAN is trained on the OCTMNIST subset of MedMNIST to generate 28×28 greyscale retinal OCT images across four classes.

Relevant notebook:

- `notebooks/part2_1_octmnist_dcgan.ipynb`

Example outputs:

- Sample OCTMNIST retinal images.
- OCTMNIST class distribution.
- DCGAN training-loss plot.
- Generated OCT-like image grids.

#### 2. CICIDS tabular GAN

A fully connected GAN is applied to network-flow features from the CICIDS 2017 Wednesday working-hours dataset, focusing on benign and DoS/DDoS traffic patterns.

Relevant notebook:

- `notebooks/part2_2_cicids_gan.ipynb`

Example outputs:

- GAN training-loss plot on CICIDS Wednesday data.
- PCA comparison of real vs generated feature vectors.
- Feature-level comparisons showing distribution mismatch and invalid generated values.

#### 3. Quick, Draw! birthday cake DCGAN

A TensorFlow DCGAN is trained on the birthday cake category from the Quick, Draw! dataset to generate simple hand-drawn sketch bitmaps.

Relevant notebook:

- `notebooks/part2_3_quickdraw_dcgan.ipynb`

Example outputs:

- Sample real birthday cake sketches.
- Generated cakes at different epochs.
- Real-vs-fake comparison panels.
- Lightweight pixel-statistics comparison.

## Data sources

During local development, the main dataset files were stored under the following project paths:

- `data/quickdraw/birthday_cake.npy`
- `data/cicids/Wednesday-workingHours.pcap_ISCX.csv`
- `data/raw/octmnist.npz`

Original or reference dataset sources:

- **MedMNIST / OCTMNIST**
  - Website: <https://medmnist.com>
  - Paper: Yang, J. et al. (2021), *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*.

- **CICIDS 2017**
  - Official dataset page: <https://www.unb.ca/cic/datasets/ids-2017.html>
  - Kaggle mirror used for convenient CSV access: <https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/data>

- **Quick, Draw! dataset**
  - GitHub documentation: <https://github.com/googlecreativelab/quickdraw-dataset>
  - Background blog post: <https://cloud.google.com/blog/products/gcp/drawings-in-the-cloud-introducing-the-quick-draw-dataset>

## Large files note

Some datasets used in this project are large and are not ideal for storing directly in a standard Git repository.

Examples include:

- `data/cicids/Wednesday-workingHours.pcap_ISCX.csv` (~215 MB locally)
- `data/quickdraw/birthday_cake.npy`

For that reason:

- Large dataset files were primarily used **locally** during experimentation.
- Not all dataset files are uploaded to GitHub in full.
- Git LFS is configured for `.csv` files through `.gitattributes`, so selected large CSV assets can be tracked with Git LFS where quota and remote limits allow.

If you clone this repository, you may need to download the original datasets separately and place them in the same `data/` folder structure.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using macOS/Linux and already have a local environment set up, you can reuse it as long as the required packages from `requirements.txt` are installed.

## Running the notebooks

Suggested order:

1. `notebooks/part1_toy_2d_gans.ipynb`
2. `notebooks/part2_1_octmnist_dcgan.ipynb`
3. `notebooks/part2_2_cicids_gan.ipynb`
4. `notebooks/part2_3_quickdraw_dcgan.ipynb`

This progression mirrors the assignment structure: start with simple toy GANs, then move to image, tabular, and sketch domains.

## Key findings

High-level findings from the case study:

- Simple GANs can learn low-dimensional synthetic distributions and are useful for understanding instability, mode coverage, and architecture effects.
- DCGANs work reasonably well for low-resolution image-like data such as OCTMNIST and Quick, Draw! sketches under modest CPU-only training budgets.
- A basic fully connected GAN struggles on CICIDS tabular data because the feature space is high-dimensional, skewed, and constrained by real-world semantics.
- Loss curves alone are not enough to evaluate synthetic data quality; visual inspection, PCA structure, and feature-level checks were also necessary.

## References

- Goodfellow, I. et al. (2014). *Generative Adversarial Nets*. NeurIPS. <https://arxiv.org/abs/1406.2661>
- Yang, J. et al. (2021). *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. <https://arxiv.org/abs/2110.14795>
- Canadian Institute for Cybersecurity. *Intrusion Detection Evaluation Dataset (CIC-IDS2017)*. <https://www.unb.ca/cic/datasets/ids-2017.html>
- Google Creative Lab. *Quick, Draw! dataset*. <https://github.com/googlecreativelab/quickdraw-dataset>
- Google Cloud Blog. *Drawings in the Cloud: introducing the Quick, Draw! dataset*. <https://cloud.google.com/blog/products/gcp/drawings-in-the-cloud-introducing-the-quick-draw-dataset>

## License

See `LICENSE` for license information.
