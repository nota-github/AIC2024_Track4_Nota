# [CVPRW 2024] Road Object Detection Robust to Distorted Objects at the Edge Regions of Images

The official resitory for 8th NVIDIA AI City Challenge (Track4: Road Object Detection in Fish-Eye Cameras) from team Netspresso (Nota Inc.).

<img width=540 src="https://github.com/nota-github/AIC2024_Track4_Nota/assets/129830699/06a5578a-39d4-4492-a6c0-f88df4327266">

# Road Object Detection in Fish-Eye Cameras
We use [Co-DETR](https://github.com/nota-github/AIC2024_Track4_Nota/tree/main/Co-DETR) for detection baseline repository.
## Installation
```
# git clone this repository
git clone https://github.com/nota-github/AIC2024_Track4_Nota.git
cd AIC2024_Track4_Nota

# Build a docker container
docker build -t aic2024_track4_nota .
docker run --name aic2024_track4_nota_0 --gpus '"device=0,1,2,3,4,5,6,7"' --shm-size=8g -it -v path_to_local_repository/:/AIC2024_Track4_Nota aic2024_track4_nota

# Install Co-DETR dependencies
cd Co-DETR
pip install -v -e .
pip install fvcore einops albumentations ensemble_boxes
cd sahi
pip install -e ."[dev]" # Refer to the original repo
```

## Prepare Datasets
Download training and challenge dataset.
- [Fisheye8K](https://github.com/MoyoG/FishEye8K) dataset into Co-DETR/data/aicity_images
- [AI CITY test set](https://www.aicitychallenge.org/2024-data-and-evaluation/) into Co-DETR/data/aicity_images

```
AIC2024_Track4_Nota
    |── Co-DETR
        |──	data
           |──	aicity_images
           |──	aicity_images_sr # 2.0x upscaled AICITY test images(using SR)
           |──	Fisheye8K
                |──	test
                |   |── images
                |   |── test_lvis.json
                |   |── test.json
                |──	train
                    |── images
                    |── train_lvis.json
                    |── train_sr.json
                    |── train.json
       ...
```
<img src="https://github.com/nota-github/AIC2024_Track4_Nota/assets/129830699/c0ce223e-ec8a-4e00-9627-5613ec91cf63">  

- We use a semi-supervision dataset(background labels from the LVIS dataset) and an upscaled SR dataset. Each JSON file can be downloaded from [drive](https://drive.google.com/drive/folders/13T1npW44v9DJUphGEwfNnh027Phih8Wk?usp=drive_link).


## Inference
Download checkpoints from the googledrive. 
- ViT-l backbone [download](https://drive.google.com/file/d/1gL7q5Cr-_4ZbVJrw4YUu3BAr4AFpNgHE/view?usp=drive_link)
```
# Co-DINO (ViT-L) + SAHI
python demo/submit_demo_sahi.py data/aicity_images \ 
    projects/configs/AIC24/co_dino_5scale_vit_large_fisheye.py \
    co_dino_5scale_vit_large_fisheye.pth \
    --out-file output \
    --device cuda:0 \
    --dataset fisheye8k \ 
    --use_hist_equal {True or False}
```

- ViT-l backbone + basic augmentation [download](https://drive.google.com/file/d/1v2N76F2nUiK3CItmbsdilTqg1JGsWIwx/view?usp=drive_link)
```
# Co-DINO (ViT-L) + basic augmentation + SAHI
python demo/submit_demo_sahi.py data/aicity_images \ 
    projects/configs/AIC24/co_dino_5scale_vit_large_fisheye_basic_aug.py \
    co_dino_5scale_vit_large_fisheye_basic_aug.pth \
    --out-file output \
    --device cuda:0 \
    --dataset fisheye8k \
    {--use_hist_equal} # If use histogram equalization
```
- ViT-l backbone + rotation augmentation [download](https://drive.google.com/file/d/1Oc9E_YYY4EZ-85PbiTsB_7nB8lIehcbi/view?usp=drive_link)
```
# Co-DINO (ViT-L) + image rotation + SAHI
python demo/submit_demo_sahi.py data/aicity_images \ 
    projects/configs/AIC24/co_dino_5scale_vit_large_fisheye_rotate.py \
    co_dino_5scale_vit_large_fisheye_rotate.pth \
    --out-file output_rotate \
    --device cuda:0 \
    --dataset fisheye8k \ 
    {--use_hist_equal} # If use histogram equalization
```
- Swin backbone + semi-supervision [download](https://drive.google.com/file/d/1msSj_hFMcLZ_e2JJEKQyB9F6wP5tSAZU/view?usp=drive_link)
```
# Co-DINO (Swin-L) + image rotation + semi-supervision + SAHI
python demo/submit_demo_sahi.py data/aicity_images \ 
    projects/configs/AIC24/co_dino_swin_fisheye8k_lvis_add_ann.py \
    co_dino_swin_fisheye8k_lvis_add_ann.pth \
    --out-file output_lvis \
    --device cuda:0 \
    --dataset fisheye8klvis \ 
    {--use_hist_equal} # If use histogram equalization
```
- ViT-l backbone + SR [download]()
```
# Co-DINO (ViT-L) + SR + SAHI
python demo/submit_demo_sahi.py data/aicity_images_sr \ 
    projects/configs/AIC24/co_dino_5scale_vit_large_fisheye_sr.py \
    co_dino_5scale_vit_large_fisheye_sr.pth \
    --out-file output_sr \
    --device cuda:0 \
    --dataset fisheye8k \
    --use_super_resolution True \
    --aicity_test_images_dir data/aicity_images
```

## Ensemble
- We use weighted boxes fusion(WBF) for ensemble. And We ensembled a total of 9 output json files.
    - Co-DINO (ViT-L) + SAHI
    - Co-DINO (ViT-L) + basic augmentation + SAHI
    - Co-DINO (ViT-L) + image rotation + SAHI
    - Co-DINO (Swin-L) + image rotation + semi-supervision + SAHI
    - Co-DINO (ViT-L) + SAHI + histogram equalization
    - Co-DINO (ViT-L) + basic augmentation + SAHI + histogram equalization
    - Co-DINO (ViT-L) + image rotation + SAHI + histogram equalization
    - Co-DINO (Swin-L) + image rotation + semi-supervision + SAHI + histogram equalization
    - Co-DINO (ViT-L) + SR + SAHI

```
python ensemble.py 
    --test_dataset_path data/aicity_images \
    --target_json_dir path_to_json_dir # The path of the dir containing the above 9 output json files is
    --out_name ensemble.json \
    --iou_thr 0.4 \
    --score_thr 0.4    
```
## Training
- Prepare pre-trained checkpoint from original Co-DETR repository.
    - For Co-DETR with ViT-Large checkpoint, refer to [this](https://github.com/Sense-X/Co-DETR/blob/main/docs/en/sota_release.md)
- Modify the config file and enter the appropriate dataset path (refer to [MMDetection's official instructions](https://mmdetection.readthedocs.io/en/latest/user_guides/tracking_config.html#learn-about-configs)).
```
bash tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
    
# Example
bash tools/dist_train.sh \
    projects/configs/AIC24/co_dino_5scale_vit_large_fisheye.py \
    8 \
    work_dirs/vit_l
```

### Super Resolution
- We utilize super-resolution (SR) technique to obtain high-resolution images for training and testing by using pre-trained [StableSR](https://github.com/nota-github/AIC2024_Track4_Nota/tree/main/StableSR) model.
    - Configure the environment by referring to [installation guide](https://github.com/IceClear/StableSR?tab=readme-ov-file#dependencies-and-installation) in the StableSR repository.
    - Upscale the images of the Fisheye8K dataset by using the provided pre-trained model.
    ```
    python scripts/sr_val_ddim_text_T_negativeprompt_canvas_tile.py \
        --config configs/stableSRNew/v2-finetune_text_T_768v.yaml \
        --ckpt stablesr_768v_000139.ckpt \ # Change the model if you need
        --vqgan_ckpt vqgan_cfw_00011.ckpt \
        --init-img /home/data/fisheye_train \ # Dataset image path
        --outdir ./outputs_fisheye_train/ \ 
        --ddim_steps 10 \
        --dec_w 0.5 \
        --colorfix_type wavelet \
        --scale 7.0 \
        --use_negative_prompt \
        --upscale 1.5 \
        --seed 42 \
        --n_samples 1 \
        --input_size 768 \
        --tile_overlap 48 \
        --ddim_eta 1.0 \
        --fold 0 
    ```

# Acknowledgement
- This project is based on [Co-DETR](https://github.com/Sense-X/Co-DETR.git) and [StableSR](https://github.com/IceClear/StableSR).