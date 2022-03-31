# W4995DL-Final
This is the final project of 2022 Spring W4995 012 Deep Learning in Columbia University.


## MaskGIT Model Discription
Reference: https://arxiv.org/abs/2202.04200

<p align="center">
<img width="718" alt="workflow" src="/asset/Mask.png">
</p>

## Code Reference
We adopted the MaskGIT pipeline from dome272's github repo: https://github.com/dome272/MaskGIT-pytorch, and change the VAGAN model to an ImageNet-pretrained model [VQModel](https://github.com/CompVis/taming-transformers).

The adopted pipelines are in the [MaskGIT-pytorch](/MaskGIT-pytorch/) folder. And the final excutive command could be found in [trainTransformer.ipynb](./trainTransformer.ipynb).

* Pretrained VQGAN model parameter: [vqgan_imagenet_f16_16384](./taming-transformers/logs/vqgan_imagenet_f16_16384/)

## What we do
- [x] Debug the gamma functions
- [x] Adopted pretrained VQGAN Model
- [ ] Add test class and calculate evaluation metric (FID, IS)
- [ ] Train on the entire ImageNet dataset
- [ ] Adopt to audio/video data

## Results
### Image generation
<p align="center">
<img width="718" alt="workflow" src="/results/39.jpg">
</p>

### Model saved

[transformer_epoch_30.pt](./checkpoints/)

