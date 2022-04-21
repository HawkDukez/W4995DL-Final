# W4995DL-Final
This is the final project of 2022 Spring W4995 012 Deep Learning in Columbia University.


## MaskGIT Model On Audio Generation
Reference: https://arxiv.org/abs/2202.04200

<p align="center">
<img width="718" alt="workflow" src="/asset/Mask.png">
</p>

## Code Reference
We adopted the MaskGIT pipeline from dome272's github repo: https://github.com/dome272/MaskGIT-pytorch, and change the VAGAN model to an ImageNet-pretrained model [VQModel](https://github.com/CompVis/taming-transformers) for image generation, and an VAS vedio spectrum-pretrained model [SpecVQGAN](https://github.com/v-iashin/SpecVQGAN).

### MaskGIT on images
* The adopted pipelines folder: [image_model](/image_model/) 
* The final excutive command: [trainTransformer.ipynb](./image_model/trainTransformer.ipynb)

* Pretrained VQGAN model parameter: [vqgan_imagenet_f16_16384](./image_model/taming-transformers/logs/vqgan_imagenet_f16_16384/)

### MaskGIT on audios
* The adopted pipelines folder: [audio_model](/audio_model/) 
* The final excutive command: [training.ipynb](./audio_model/training.ipynb)

Model training code:
```bash
python ./training_transformer.py --batch-size=8 --num-codebook-vectors=265 --epochs=40 --image-channels=1`
```

## What we do
#### Stage 1: image model regeneration
- [x] Debug the gamma functions
- [x] Adopt pretrained VQGAN Model
- [x] Add test class and calculate evaluation metric (FID, IS)
- [ ] Train on the entire ImageNet dataset

#### Stage 2: Model adoptation to audio data
- [x] Find proper audio data (VAS)
- [x] Adopt MaskGIT transformer to the pretrained SpecVQGAN model
- [x] Compare the difference and efficacy of two models
- [ ] Train on the entire VAS and VGGSound dataset


## Results
### Image generation
<p align="center">
<img width="718" alt="workflow" src="/results/39.jpg">
</p>

### Model saved

[transformer_epoch_30.pt](./checkpoints/)

### Audio generation


