# Anatomy-Anchored Self-Supervision: Distilling Vision Foundation Models for Invariant Ultrasound Representation

## Abstract

Self-supervised pre-training has gained increasing prominence for learning transferable representations in medical imaging, yet existing methods for ultrasound (US) images operate at the image or frame level, overlooking the anatomical context for clinical-aligned representation learning. We propose **ANAUS**, an **Ana**tomy-Anchored **U**ltra**S**ound Self-Supervision framework that shifts representation learning from generic visual regions to clinically meaningful anatomical structures. Utilizing a learnable latent prompt engine alongside a one-time domain adaptation on existing public image–mask pairs, we empower the LP-SAM module to achieve annotation-free anatomy delineation at scale. Building upon this anatomical grounding, ANAUS integrates two synergistic learning mechanisms: (i) inter-view semantics-aware anatomy-separating alignment, and (ii) contextual core-region prediction. Extensive evaluations on six public datasets demonstrate that ANAUS consistently outperforms current state-of-the-art methods while maintaining the computational efficiency essential for clinical deployment.

![ANAUS Framework](./figs/ANAUS.png)

## 🔨 PostScript

😄 This project is the PyTorch implementation of ANAUS.

😆 Our experimental platform is configured with two *RTX 4090* GPUs (24 GB VRAM each).

## 💻 Installation

1. Clone or download this repository.

   ```bash
   cd <ANAUS_project_dir>
   ```

2. Create conda environment.

   ```bash
   conda create -n ANAUS python=3.9
   conda activate ANAUS
   ```

3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

## 🐾 ANAUS Evaluation

1. Load the pre-trained ANAUS weights from `./checkpoint/pre-trained-anaus.pth`.

2. Download the downstream evaluation datasets:
   - **Classification**: [POCUS](https://drive.google.com/file/d/1w7FrwqQ09VjwtTcZL5M0hZnW3Oly9Buv/view?usp=drive_link), BUSI
   - **Segmentation**: UDIAT-B, TN3K, BUSI, HMC-QU

   All downstream tasks use five-fold cross-validation.

3. Run the evaluation with:

   ```bash
   python ./evaluation_anaus.py --data_path <data_path> --ckpt_path <pre_trained_anaus_path>
   ```

## 📘 Fine-tune LP-SAM

### Checkpoint

Download the checkpoint for SAM (Segment Anything Model): [ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

### Dataset

1. Download the paired image–mask datasets used for LP-SAM adapter fine-tuning:
   - [DDTI](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation) (637 images)
   - [TG3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation) (3,585 images)
   - [CAMUS](http://camus.creatis.insa-lyon.fr/challenge/) (57,696 images)

   Convert all datasets to `.png` format and organize them as follows:

   ```none
   dataset
   ├── ThyroidGland-DDTI
   │   ├── img
   │   │   ├── xxx.png
   │   │   └── ...
   │   └── label
   │       ├── xxx.png
   │       └── ...
   ├── Echocardiography-CAMUS
   │   ├── img
   │   │   ├── xxx.png
   │   │   └── ...
   │   └── label
   │       ├── xxx.png
   │       └── ...
   └── MainPatient
   ```

2. The `./SAM/MainPatient` folder contains `train/val.txt` with lines formatted as:

   ```
   <class ID>/<dataset folder name>/<image file name>
   ```

3. Set other configurations in `./SAM/utils/config.py`.

### Stage 1: Adapter-based Domain Specialization

Fine-tune the SAM adapter for the ultrasound domain:

```bash
python ./LPSAM/only_train_sam.py --sam_ckpt <sam_vit_b.pth>
```

### Stage 2: Latent Prompt Engine Training

Fine-tune the latent prompt engine with SAM frozen:

```bash
python ./LPSAM/train_lpe.py --sam_ckpt <finetuned_sam.pth> --load_auto_prompter
```

> **Note:** Both LP-SAM stages are one-time procedures and are excluded from subsequent SSL pre-training iterations.

## 🐾 ANAUS Pre-training

1. Download the SSL pre-training datasets ([Butterfly](https://www.butterflynetwork.com/) and [CAMUS](http://camus.creatis.insa-lyon.fr/challenge/)), then generate anatomical masks via LP-SAM inference. Organize the directory as follows:

   ```none
   dataset
   ├── train
   │   ├── Butterfly
   │   │   ├── img
   │   │   └── label
   │   └── CAMUS
   │       ├── img
   │       └── label
   ```

2. Run `./utils/generate_masks_pkls.py` to generate mask `.pkl` files and index file:

   ```bash
   python ./utils/generate_masks_pkls.py
   ```

   The resulting directory structure should be:

   ```none
   dataset
   ├── train
   │   ├── Butterfly
   │   │   ├── img
   │   │   └── label
   │   └── CAMUS
   │       ├── img
   │       └── label
   ├── masks
   │   ├── xxx.pkl
   │   └── ...
   └── train_tf_img_to_gt.pkl
   ```

3. Set training options in the config file `./configs/example.yaml`. Assign `resume_path` to the checkpoint you wish to resume from if applicable.

4. Launch distributed pre-training with:

   ```bash
   python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 pretrain_anaus.py --cfg ./configs/example.yaml
   ```