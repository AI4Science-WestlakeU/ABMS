# Additional Backward Guided Diffusion

The code of handwriting generation task is based on [[paper]](https://arxiv.org/abs/2402.03201).

The code of both Linear Inverse Problem and Non-linear Problem are based on **DSG** [[paper]](https://arxiv.org/abs/2402.03201).

## Linear Inverse Problem

### 1) Set environment

Install dependencies:

```
cd linear

conda create -n DSG python=3.8

conda activate DSG

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

```

### 2) Download checkpoints

Download checkpoint `ffhq_10m.pt` or `imagenet256.pt`  from guidance of DSG's official repository and paste it to `./linear/models/`. 

### 3) dataset preparation

You could use the script `./linear/get_ffhq_data.py` or `./linear/get_imagenet_data.py` to download the dataset from huggingface. Find your ffhq-256 or imagenet-256 dataset name on huggingface and replace the `HUGGINGFACE_DATASET_NAME` in the script.

```
cd linear
python get_ffhq_data.py
# or
python get_imagenet_data.py
```

### 4) Generate noisy measurement

You could modify the parameters following the comment in `./linear/scripts/generate_ffhq.sh` or `./linear/scripts/generate_imagenet.sh` and run it.

```
cd linear
bash scripts/generate_ffhq.sh
# or
bash scripts/generate_imagenet.sh
```

### 5) Inference

You could modify the parameters following the comment in `./linear/run_DSG.sh` and run it using the hyperparameter in Table 3 in the Appendix of paper. 
The results are shown in `./linear/total_results_DSG_DDIM"$DDIM"/DSG_interval_${interval}_ guidance_${guidance_scale}/{TASK}/recon`.

```
cd linear
bash run_ffhq.sh
# or
bash run_imagenet.sh
```

### 6) Evaluation in FFHQ/Imagenet

Change the **RESULT_GT_PAIRS** in `./linear/scripts/eval.sh` and run it.

```
cd linear
bash scripts/eval.sh
```

## Non-linear Inverse Problem

### 1) Set environment

Install dependencies:

```bash
cd non-linear

apt-get install libsm6
apt-get install libxrender1
apt-get install libxext-dev

conda env create -f environment.yaml
conda activate ldm
```
### 2) Download checkpoints

#### 2.1) FaceID Guidance

Download pretrained model  `celebahq.ckpt` and `model_ir_se50.pth` following guidance of DSG's official repository
and place them to `./non-linear/Face-GD/exp/celebahq.ckpt` and `./non-linear/Face-GD/exp/model_ir_se50.pth`.

#### 2.2) Style/Text-Style Guidance

Download pretrained model SD-v1-4 following guidance of DSG's official repository and place it to `./non-linear/SD_style/models/ldm/stable-diffusion-v1/model.ckpt`.

### 3) Quick Start

#### 3.1) FaceID Guidance

`cd non-linear/Face-GD/` and run `bash run_faceid_ffhq.sh`.

#### 3.2) Style Guidance

Place the 256x256 style image you want to guide in `./non-linear/SD_style/style_images/`.

`cd non-linear/SD_style/` and run `bash run_style_guidance.sh`.

#### 3.3) Text-Style Guidance

Place the 256x256 image you want to guide in text-style guidance in `./non-linear/SD_style/text_style_images/`.

`cd non-linear/SD_style/` and run `bash run_text_style_guidance.sh`.

### 4) Evaluation
For FaceID Guidance:
`cd non-linear/Face-GD/` modify the **real_folder** and **gen_folder** in `bash run_eval.sh` and run `bash run_eval.sh`.
