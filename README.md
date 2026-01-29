# VAE-GAN-anime-face-generator
Variational Autoencoder and GAN trained on kaggle anime face dataset for character generation and interpolation experiments.

## Overview
This project implements a **Variational Autoencoder (VAE)** and a **Generative Adversarial Network (GAN)** to generate high-quality anime faces.  
The VAE allows for **image reconstruction** and **latent space interpolation**, while the GAN produces **visually realistic samples**.  
This project demonstrates skills in **deep learning, generative modeling, PyTorch, and image preprocessing**.

---

## Features
- Train a **VAE** on anime face dataset for reconstruction and interpolation
- Generate new images from the **VAE latent space**
- Train a **GAN** (optional) for high-quality sample generation
- Save outputs as image grids for easy visualization
- Includes checkpointing and reproducible experiments

---

## Dataset
- **Anime Faces Dataset** from [Kaggle]((https://www.kaggle.com/datasets/splcher/animefacedataset/data))  
- **License:** Database Contents License (DbCL) v1.0 (Open Data Commons)  
- **Note:** Dataset is **not included** in this repository.  
  Please download and place it in a folder named `data/` at the root of the repo.

```bash
anime-face-generator/
└── data/  # contains dataset images organized in subfolders
