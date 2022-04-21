"""
################################################
## The utils codes is adopted from:
## https://github.com/dome272/MaskGIT-pytorch
## https://github.com/v-iashin/SpecVQGAN
##                     
## Modifications:
## 1. load_data function modified to audio input
## 2. 
################################################
"""

import os
import shutil
import albumentations
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm
from specvqgan.data.vggsound import CropImage
from glob import glob
from sample_visualization import (load_feature_extractor,
                                  load_model_from_config, load_vocoder)
from specvqgan.data.vggsound import CropFeats
from specvqgan.util import download, md5_hash
from specvqgan.models.cond_transformer import disabled_train
from train import instantiate_from_config

from feature_extraction.extract_mel_spectrogram import get_spectrogram


# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

class ImagePaths(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        #self.images = [os.path.join(path, file) for file in os.listdir(path)]
        #self.image_dir = [os.path.join(path,directory) for directory in os.listdir(path)]
        #self.images = np.load(path)
        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

#         self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
#         self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
#         self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
#         image = Image.open(image_path)
#         if not image.mode == "RGB":
#             image = image.convert("RGB")
        image = np.load(image_path)
        image = {'input': image}
        
        random_crop = False
        crop_img_fn = CropImage([80, 848], random_crop)
        image = crop_img_fn(image)
        # Prepare input
#         batch = default_collate([image])
#         batch['image'] = batch['input'].to(device)
#         x = sampler.get_input(batch,sampler.first_stage_key)
        
#         shape = image.shape
#         image = iamge.reshape([1,shape[0],shape[1]])

#         image = Image.fromarray(np.uint8(cm.gist_earth(image)*255))
#         if not image.mode == "RGB":
#              image = image.convert("RGB")
#         image = np.array(image).astype(np.uint8)
#         image = self.preprocessor(image=image)["image"]
#         image = (image / 127.5 - 1.0).astype(np.float32)
#         image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example


def load_data(args):
    train_data = ImagePaths(args.dataset_path, size=256)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    return train_loader


def load_testdata(args):
    test_data = ImagePaths(args.test_dataset_path, size=256)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    return test_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images: dict):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    new_sample = images["new_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(new_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()


# --------------------------------------------- #
#               Model Loading Utils
# --------------------------------------------- #
    
def maybe_download_model(model_name: str, log_dir: str) -> str:
    name2info = {
        '2021-06-20T16-35-20_vggsound_transformer': {
            'info': 'No Feats',
            'hash': 'b1f9bb63d831611479249031a1203371',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-06-20T16-35-20_vggsound_transformer.tar.gz',
        },
        '2021-07-30T21-03-22_vggsound_transformer': {
            'info': '1 ResNet50 Feature',
            'hash': '27a61d4b74a72578d13579333ed056f6',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-07-30T21-03-22_vggsound_transformer.tar.gz',
        },
        '2021-07-30T21-34-25_vggsound_transformer': {
            'info': '5 ResNet50 Features',
            'hash': 'f4d7105811589d441b69f00d7d0b8dc8',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-07-30T21-34-25_vggsound_transformer.tar.gz',
        },
        '2021-07-30T21-34-41_vggsound_transformer': {
            'info': '212 ResNet50 Features',
            'hash': 'b222cc0e7aeb419f533d5806a08669fe',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-07-30T21-34-41_vggsound_transformer.tar.gz',
        },
        '2021-06-03T00-43-28_vggsound_transformer': {
            'info': 'Class Label',
            'hash': '98a3788ab973f1c3cc02e2e41ad253bc',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-06-03T00-43-28_vggsound_transformer.tar.gz',
        },
        '2021-05-19T22-16-54_vggsound_codebook': {
            'info': 'VGGSound Codebook',
            'hash': '7ea229427297b5d220fb1c80db32dbc5',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-05-19T22-16-54_vggsound_codebook.tar.gz',
        },
        '2021-06-06T19-42-53_vas_codebook': {
            'info': 'VAS Codebook',
            'hash': '0024ad3705c5e58a11779d3d9e97cc8a',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-06-06T19-42-53_vas_codebook.tar.gz',
        },
        '2021-06-20T16-24-38_vas_transformer': {
            'info': 'VAS Transformer',
            'hash': 'ea4945802094f826061483e7b9892839',
            'link': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a'
                    '/specvqgan_public/models/2021-06-20T16-24-38_vas_transformer.tar.gz',
        }
    }
    print(f'Using: {model_name} ({name2info[model_name]["info"]})')
    model_dir = os.path.join(log_dir, model_name)
    if not os.path.exists(model_dir):
        tar_local_path = os.path.join(log_dir, f'{model_name}.tar.gz')
        # check if tar already exists and its md5sum
        if not os.path.exists(tar_local_path) or md5_hash(tar_local_path) != name2info[model_name]['hash']:
            down_link = name2info[model_name]['link']
            download(down_link, tar_local_path)
            print('Unpacking', tar_local_path, 'to', log_dir)
            shutil.unpack_archive(tar_local_path, log_dir)
            # clean-up space as we already have unpacked folder
            os.remove(tar_local_path)
    return model_dir

def load_config(model_dir: str):
    # Load the config
    config_main = sorted(glob(os.path.join(model_dir, 'configs/*-project.yaml')))[-1]
    config_pylt = sorted(glob(os.path.join(model_dir, 'configs/*-lightning.yaml')))[-1]
    config = OmegaConf.merge(
        OmegaConf.load(config_main),
        OmegaConf.load(config_pylt),
    )
    # patch config. E.g. if the model is trained on another machine with different paths
    for a in ['spec_dir_path', 'rgb_feats_dir_path', 'flow_feats_dir_path']:
        if config.data.params[a] is not None:
            print(config.data.params.train.target)
            if 'vggsound.VGGSound' in config.data.params.train.target:
                base_path = './data/vggsound/'
            elif 'vas.VAS' in config.data.params.train.target:
                base_path = './data/vas/features/*/'
            else:
                raise NotImplementedError
            config.data.params[a] = os.path.join(base_path, Path(config.data.params[a]).name)
    return config

def load_model(model_name, log_dir, device):
    to_use_gpu = True if device.type == 'cuda' else False
    model_dir = maybe_download_model(model_name, log_dir)
    config = load_config(model_dir)

    # Sampling model
    ckpt = sorted(glob(os.path.join(model_dir, 'checkpoints/*.ckpt')))[-1]
    pl_sd = torch.load(ckpt, map_location='cpu')
    sampler = load_model_from_config(config.model, pl_sd['state_dict'], to_use_gpu)['model']
    sampler.to(device)

    # aux models (vocoder and melception)
    ckpt_melgan = config.lightning.callbacks.image_logger.params.vocoder_cfg.params.ckpt_vocoder
    melgan = load_vocoder(ckpt_melgan, eval_mode=True)['model'].to(device)
    melception = load_feature_extractor(to_use_gpu, eval_mode=True)
    return config, sampler, melgan, melception


