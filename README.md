# Additional Backward Guided Diffusion

The code of Linear Inverse Problem and Non-linear Problem are based on **DSG** [[paper]](https://arxiv.org/abs/2402.03201).

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

## 3) Generate noisy measurement

You could modify the parameters following the comment in `./linear/scripts/generate_ffhq.sh` or `./linear/scripts/generate_imagenet.sh` and run it.

```
cd linear
bash scripts/generate_ffhq.sh
# or
bash scripts/generate_imagenet.sh
```

### 4) Inference

You could modify the parameters following the comment in `./linear/run_DSG.sh` and run it using the hyperparameter in Table 3 in the Appendix of paper. 
The results are shown in `./linear/total_results_DSG_DDIM"$DDIM"/DSG_interval_${interval}_ guidance_${guidance_scale}/{TASK}/recon`.

```
cd linear
bash run_ours_ffhq.sh
# or
bash run_ours_imagenet.sh
```

### 5) Evaluation in FFHQ/Imagenet

Change the **RESULT_GT_PAIRS** in `./linear/scripts/eval.sh` and run it.

```
cd linear
bash scripts/eval.sh
```




