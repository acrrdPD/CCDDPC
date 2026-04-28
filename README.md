<div align="center">
<h1>Cross-Modal Compensation and Dual Deformation Prediction Correction: Breaking Barriers in Unaligned Infrared-Visible Image Fusion</h1>

<!-- Replace the following title with the official paper title if needed. -->
<h3>Code for "Cross-Modal Compensation and Dual Deformation Prediction Correction: Breaking Barriers in Unaligned Infrared-Visible Image Fusion"</h3>

[**Paper**](#) | [**Code**](https://github.com/acrrdPD/CCDDPC) | [**Models**](#)

<!-- <sup>*</sup>corresponding authors -->

<!-- DOI badge can be added after the paper is officially online. -->
<!-- <a href='YOUR_DOI_LINK'><img src='https://img.shields.io/badge/DOI-XXXX-blue'></a> -->

</div>

**CCDDPC** provides the implementation of our infrared and visible image registration and fusion framework. The project contains training and testing scripts for feature extraction, image registration/alignment, and image fusion.


## 📢 News
- **2026-04-28** 🔥 The initial code repository is available.
- Pre-trained models and detailed instructions will be updated after publication.


## 📝 TODOs
- [x] Release initial code
- [ ] Release pre-trained models
- [ ] Release detailed dataset preparation instructions
- [ ] Add paper DOI and BibTeX citation


## ✨ Usage

### Quick start

#### 1. Clone this repository and set up the environment

```sh
git clone https://github.com/acrrdPD/CCDDPC.git
cd CCDDPC

conda create -n CCDDPC python=3.8 -y
conda activate CCDDPC
```

Install the required packages according to your CUDA and PyTorch versions. A basic environment can be prepared as follows:

```sh
pip install numpy tqdm matplotlib kornia setproctitle opencv-python pillow
pip install ftfy regex einops
```

Please install PyTorch from the official website according to your CUDA version:

```sh
# Example only. Please modify it according to your CUDA version.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If your environment does not include CLIP, please install it by:

```sh
pip install git+https://github.com/openai/CLIP.git
```


#### 2. Prepare data

Please organize the training and testing data according to the paths used in the training and testing scripts. The default training path in the current code follows the form:

```markdown
📦 dataset
├── 📂 Train
│   └── 📂 512RS
│       ├── 📂 ir          # Infrared images
│       ├── 📂 vi          # Visible images
│       ├── 📂 ir_map      # Infrared map / auxiliary map
│       └── 📂 vi_map      # Visible map / auxiliary map
└── 📂 Test
    ├── 📂 ir              # Infrared images
    └── 📂 vi              # Visible images
```

If your dataset is stored in another location, please modify the corresponding paths in the scripts under `Trainer/` and `Test/`.


#### 3. Download pre-trained models

The pre-trained models will be released after publication.

| Model | Download Link | Description |
| :--- | :--- | :--- |
| CCDDPC | Coming soon | Pre-trained model for infrared and visible image registration and fusion |

After downloading the models, please place them in the corresponding checkpoint directory, or modify the checkpoint paths in the scripts.


#### 4. Test

To test the model, please modify the dataset path and checkpoint path in the testing script, and then run:

```sh
python Test/test_align_fusion_s.py
```

The testing results will be saved to the output directory specified in the script.


## ⚙️ Training

The project contains multiple training scripts for different stages.

### 1. Train the fusion module

```sh
python Trainer/train_fusion.py
```

### 2. Train the alignment/registration module

```sh
python Trainer/train_align_t.py
```

### 3. Train the alignment-fusion framework

```sh
python Trainer/train_align_fusion_s.py
```

Before training, please carefully check the following settings in each script:

```python
--ir        # path to infrared training images
--vis       # path to visible training images
--ir_map    # path to infrared auxiliary maps
--vis_map   # path to visible auxiliary maps
--ckpt_e    # checkpoint of the feature extraction module
--ckpt_f    # checkpoint of the fusion module
--ckpt_affine # checkpoint of the registration/alignment module
```


## 📁 Project Structure

```markdown
📦 CCDDPC
├── 📂 Trainer               # Training scripts
│   ├── train_fusion.py
│   ├── train_align_t.py
│   └── train_align_fusion_s.py
├── 📂 Test                  # Testing scripts
├── 📂 models                # Network architectures
├── 📂 dataloader            # Dataset loading and preprocessing
├── 📂 loss                  # Loss functions
├── 📂 functions             # Transformation and utility functions
├── 📂 clip                  # CLIP-related code
├── 📂 data                  # Data processing scripts
├── 📂 util                  # Utility functions
└── 📂 dataset               # Dataset directory
```


## 💡 Notes

1. Please make sure that the CUDA version, PyTorch version, and GPU driver are compatible.
2. Please check the dataset paths and checkpoint paths before running the scripts.
3. Large files such as datasets, checkpoints, and experimental results are recommended to be stored outside the Git repository.
4. If you meet path-related errors, please first check whether the folder names are consistent with the paths defined in the scripts.


## 👏 Acknowledgment

This repository is built upon many excellent open-source projects and research works. We sincerely thank the authors and contributors of these projects.


## 📬 Contact

For questions about the code or paper, please open an issue in this repository or contact the authors.


## 🎓 Citation

If this code is helpful to your research, please cite our paper. The BibTeX entry will be updated after the paper is officially published.

```bibtex
@article{CCDDPC,
  title   = {Cross-Modal Compensation and Dual Deformation Prediction Correction: Breaking Barriers in Unaligned Infrared-Visible Image Fusion},
  author  = {Keying Du, Yafei Zhang, Huafeng Li, Zhengtao Yu, Yu Liu},
  journal = {IEEE Transactions on Emerging Topics in Computational Intelligence},
  year    = {2026},
  note    = {To appear}
}
```
