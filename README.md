# WDM: 3D Wavelet Diffusion Models for High-Resolution Medical Image Synthesis
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-page-blue)](https://pfriedri.github.io/wdm-3d-io/)
[![arXiv](https://img.shields.io/badge/arXiv-2402.19043-b31b1b.svg)](https://arxiv.org/abs/2402.19043)

This is the official PyTorch implementation of the paper **WDM: 3D Wavelet Diffusion Models for High-Resolution Medical Image Synthesis** by [Paul Friedrich](https://pfriedri.github.io/), [Julia Wolleb](https://dbe.unibas.ch/en/persons/julia-wolleb/), [Florentin Bieder](https://dbe.unibas.ch/en/persons/florentin-bieder/), [Alicia Durrer](https://dbe.unibas.ch/en/persons/alicia-durrer/) and [Philippe C. Cattin](https://dbe.unibas.ch/en/persons/philippe-claude-cattin/).


If you find our work useful, please consider to :star: **star this repository** and :memo: **cite our paper**:
```bibtex
@inproceedings{friedrich2024wdm,
               title={Wdm: 3d wavelet diffusion models for high-resolution medical image synthesis},
               author={Friedrich, Paul and Wolleb, Julia and Bieder, Florentin and Durrer, Alicia and Cattin, Philippe C},
               booktitle={MICCAI Workshop on Deep Generative Models},
               pages={11--21},
               year={2024},
               organization={Springer}}
```

## Paper Abstract
Due to the three-dimensional nature of CT- or MR-scans, generative modeling of medical images is a particularly challenging task. Existing approaches mostly apply patch-wise, slice-wise, or cascaded generation techniques to fit the high-dimensional data into the limited GPU memory. However, these approaches may introduce artifacts and potentially restrict the model's applicability for certain downstream tasks. This work presents WDM, a wavelet-based medical image synthesis framework that applies a diffusion model on wavelet decomposed images. The presented approach is a simple yet effective way of scaling diffusion models to high resolutions and can be trained on a single 40 GB GPU. Experimental results on BraTS and LIDC-IDRI unconditional image generation at a resolution of 128 x 128 x 128 show state-of-the-art image fidelity (FID) and sample diversity (MS-SSIM) scores compared to GANs, Diffusion Models, and Latent Diffusion Models. Our proposed method is the only one capable of generating high-quality images at a resolution of 256 x 256 x 256.

<p>
    <img width="750" src="assets/wdm.png"/>
</p>


## Dependencies
We recommend using a [conda](https://github.com/conda-forge/miniforge#mambaforge) environment to install the required dependencies.
You can create and activate such an environment called `wdm` by running the following commands:
```sh
mamba env create -f environment.yml
mamba activate wdm
```

## Training & Sampling
For training a new model or sampling from an already trained one, you can simply adapt and use the script `run.sh`. All relevant hyperparameters for reproducing our results are automatically set when using the correct `MODEL` in the general settings.
For executing the script, simply use the following command:
```sh
bash run.sh
```
**Supported settings** (set in `run.sh` file):

MODE: `'training'`, `'sampling'`

MODEL: `'ours_unet_128'`, `'ours_unet_256'`, `'ours_wnet_128'`, `'ours_wnet_256'`

DATASET: `'brats'`, `'lidc-idri'`

DROPOUT: dropout probability for the U-Net (e.g., `0.1`)

## Conditional Image Synthesis / Image-to-Image Translation 
To use WDM for conditional image synthesis or paired image-to-image translation check out our repository [pfriedri/cwdm](https://github.com/pfriedri/cwdm) that implements our paper **cWDM: Conditional Wavelet Diffusion Models for Cross-Modality 3D Medical Image Synthesis**.

## Pretrained Models
We released pretrained models on [HuggingFace](https://huggingface.co/pfriedri/wdm-3d).

Currently available models:
- [BraTS 128](https://huggingface.co/pfriedri/wdm-3d/blob/main/brats_unet_128_1200k.pt): BraTS, 128 x 128 x 128, U-Net backbone, 1.2M Iterations
- [LIDC-IDRI 128](https://huggingface.co/pfriedri/wdm-3d/blob/main/lidc-idri_unet_128_1200k.pt): LIDC-IDRI, 128 x 128 x 128, U-Net backbone, 1.2M Iterations 

## Data
To ensure good reproducibility, we trained and evaluated our network on two publicly available datasets:
* **BRATS 2023: Adult Glioma**, a dataset containing routine clinically-acquired, multi-site multiparametric magnetic resonance imaging (MRI) scans of brain tumor patients. We just used the T1-weighted images for training. The data is available [here](https://www.synapse.org/#!Synapse:syn51514105).

* **LIDC-IDRI**, a dataset containing multi-site, thoracic computed tomography (CT) scans of lung cancer patients. The data is available [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254).

The provided code works for the following data structure (you might need to adapt the `DATA_DIR` variable in `run.sh`):
```
data
└───BRATS
    └───BraTS-GLI-00000-000
        └───BraTS-GLI-00000-000-seg.nii.gz
        └───BraTS-GLI-00000-000-t1c.nii.gz
        └───BraTS-GLI-00000-000-t1n.nii.gz
        └───BraTS-GLI-00000-000-t2f.nii.gz
        └───BraTS-GLI-00000-000-t2w.nii.gz  
    └───BraTS-GLI-00001-000
    └───BraTS-GLI-00002-000
    ...

└───LIDC-IDRI
    └───LIDC-IDRI-0001
      └───preprocessed.nii.gz
    └───LIDC-IDRI-0002
    └───LIDC-IDRI-0003
    ...
```
We provide a script for preprocessing LIDC-IDRI. Simply run the following command with the correct path to the downloaded DICOM files `DICOM_PATH` and the directory you want to store the processed nifti files `NIFTI_PATH`:
```sh
python utils/preproc_lidc-idri.py --dicom_dir DICOM_PATH --nifti_dir NIFTI_PATH
```

### Image Size
The `img_size` parameter of our dataset loaders defines the cubic size of the
volumes returned to the model. When the raw data is larger, it is first padded
or cropped to a cube and then downsampled to match `img_size`. For example,
datasets stored at 256³ will be downsampled to 128³ when using
`img_size=128`. Likewise, if your images are 128³ and you request
`img_size=64`, the loader will return 64³ volumes.

## Evaluation
As our code for evaluating the model performance has slightly different dependencies, we provide a second .yml file to set up the evaluation environment.
Simply use the following command to create and activate the new environment:
```sh
mamba env create -f eval/eval_environment.yml
mamba activate eval
```
### FID
For computing the FID score, you need to specify the following variables and use them in the command below:
* DATASET: `brats` or `lidc-idri`
* IMG_SIZE: `128` or `256`
* REAL_DATA_DIR: path to your real data
* FAKE_DATA_DIR: path to your generated/ fake data
* PATH_TO_FEATURE_EXTRACTOR: path to the feature extractor weights, e.g. `./eval/pretrained/resnet_50_23dataset.pt`
* PATH_TO_ACTIVATIONS: path to the location where you want to save mus and sigmas (in case you want to reuse them), e.g. `./eval/activations/` 
* GPU_ID: gpu you want to use, e.g. `0`
```sh
python eval/fid.py --dataset DATASET --img_size IMG_SIZE --data_root_real REAL_DATA_DIR --data_root_fake FAKE_DATA_DIR --pretrain_path PATH_TO_FEATURE_EXTRACTOR --path_to_activations PATH_TO_ACTIVATIONS --gpu_id GPU_ID
```
### Mean MS-SSIM
For computing the mean MS-SSIM, you need to specify the following variables and use them in the command below:
* DATASET: `brats` or `lidc-idri`
* IMG_SIZE: `128` or `256`
* SAMPLE_DIR: path to the generated (or real) data

```sh
python eval/ms_ssim.py --dataset DATASET --img_size IMG_SIZE --sample_dir SAMPLE_DIR
```
## Implementation Details for Comparing Methods
* **HA-GAN**: For implementing the paper [Hierarchical Amortized GAN for 3D High Resolution Medical Image Synthesis](https://ieeexplore.ieee.org/abstract/document/9770375), we use the publicly available [implementation](https://github.com/batmanlab/HA-GAN). We follow the implementation details presented in the original paper (Section E). The authors recommend cutting all zero slices from the volumes before training. To allow a fair comparison with other methods, we have omitted this step.
* **3D-LDM**: For implementing the paper [Denoising Diffusion Probabilistic Models for 3D Medical Image Generation](https://www.nature.com/articles/s41598-023-34341-2), we use the publicly available [implementation](https://github.com/FirasGit/medicaldiffusion). We follow the implementation details presented in the Supplementary Material of the original paper (Supplementary Table 1).
* **2.5D-LDM**: For implementing the paper [Make-A-Volume: Leveraging Latent Diffusion Models for Cross-Modality 3D Brain MRI Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-43999-5_56), we adopted the method to work for image generation. We trained a VQ-VAE (downsampling factor 4, latent dimension 32) using an implementation from [MONAI Generative](https://github.com/Project-MONAI/GenerativeModels) and a diffusion model implementation from [OpenAI](https://github.com/openai/guided-diffusion). For implementing the pseudo 3D layers, we use a script provided by the authors. To allow for image generation, we sample all slices at once - meaning that the models batch size and the dimension of the 1D convolution is equal to the number of slices in the volume to be generated.
* **3D DDPM**: For implementing a memory efficient baseline model, we use the 3D DDPM presented in the paper [Memory-Efficient 3D Denoising Diffusion Models for Medical Image Processing](https://openreview.net/forum?id=neXqIGpO-tn), and used the publicly available [implementation](https://github.com/FlorentinBieder/PatchDDM-3D). We use additive skip connections and train the model with the same hyperparameters as our models.

All experiments were performed on a system with an AMD Epyc 7742 CPU and a NVIDIA A100 (40GB) GPU.

## TODOs
We plan to add further functionality to our framework:
- [ ] Add compatibility for more datasets like MRNet, ADNI, or fastMRI
- [x] Release pre-trained models
- [ ] Extend the framework for 3D image inpainting
- [x] Extend the framework for 3D image-to-image translation ([pfriedri/cwdm](https://github.com/pfriedri/cwdm))

## Acknowledgements
Our code is based on / inspired by the following repositories:
* https://github.com/openai/guided-diffusion (published under [MIT License](https://github.com/openai/guided-diffusion/blob/main/LICENSE))
* https://github.com/FlorentinBieder/PatchDDM-3D (published under [MIT License](https://github.com/FlorentinBieder/PatchDDM-3D/blob/master/LICENSE))
* https://github.com/VinAIResearch/WaveDiff (published under [GNU General Public License v3.0](https://github.com/VinAIResearch/WaveDiff/blob/main/LICENSE))
* https://github.com/LiQiufu/WaveCNet (published under [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode))

For computing FID scores we use a pretrained model (`resnet_50_23dataset.pth`) from:
* https://github.com/Tencent/MedicalNet (published uner [MIT License](https://github.com/Tencent/MedicalNet/blob/master/LICENSE))

Thanks for making these projects open-source.
