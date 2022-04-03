#!/usr/bin/env python
# coding: utf-8
# %%
"""
################################################
## This is the model testing code with FID    ##
## and IS calculated.                         ##
## Author: Baode Gao                          ##
################################################
"""

# %%
get_ipython().system('pip install keras')
get_ipython().system('pip install tensorflow')
#pip install tensorflow==2.7


# %%
import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_data, plot_images
import sys

import matplotlib.pyplot as plt
from random import randint
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten,Lambda,Reshape
from keras.models import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import keras
from torch.utils.data import DataLoader
import torch
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize




sys.path.insert(0, '/home/ygong2832/taming-transformers')
sys.path.insert(0, '/home/ygong2832/taming-transformers/taming')
# import model
class Test:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.model.load_state_dict(torch.load('/home/ygong2832/checkpoints/transformer_epoch_39.pt'))
        self.model.eval()
        self.real,self.fake = self.test(args)
        
        numpy.save('save_real1500',self.real) 
        numpy.save('save_fake1500',self.fake) 
        #self.real = numpy.load('save_real.npy')
        #self.fake = numpy.load('save_fake.npy')
        
        #shuffle(self.real)
        #shuffle(self.fake)
        #self.images1 = self.real.astype('float32')
        #self.images2 = self.fake.astype('float32')
        #self.images1 = scale_images(self.images1, (299,299,3))
        #self.images2 = scale_images(self.images2, (299,299,3))
        shuffle(self.fake)
        
        print('loaded', self.fake.shape)
        is_avg, is_std = self.calculate_inception_score()
        print('score', is_avg, is_std)
        
        self.incV3 = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
        self.images1 =self.real
        shuffle(self.images1)
        self.images2 = self.fake
        
        print('Loaded', self.images1.shape, self.images2.shape)
        self.images1 = self.images1.astype('float32')
        self.images2 = self.images2.astype('float32')
        self.images1 = self.scale_images(images1, (299,299,3))
        self.images2 = self.scale_images(images2, (299,299,3))

        print('Scaled', self.images1.shape, self.images2.shape)

        self.images1 = preprocess_input(self.images1)
        self.images2 = preprocess_input(self.images2)

        fid = self.calculate_fid()
        print('FID: %.3f' % fid)
        
    def test(self, args):
        train_dataset = load_data(args)
        
        
        fake_lst, real_lst = numpy.empty([0,3, 256, 256]), numpy.empty([0,3, 256, 256])
        with tqdm(range(len(train_dataset))) as pbar:
            for i, imgs in zip(pbar, train_dataset):
                
                imgs = imgs.to(device=args.device)
                print(len(imgs))
                
                logits, target = self.model(imgs)
                for j in range(0,len(imgs)):
                    log, sampled_imgs = self.model.log_images(imgs[j][None])
                    real_lst = numpy.append(real_lst, log["input"].cpu().numpy(), axis=0)
                    fake_lst = numpy.append(fake_lst, log["rec"].cpu().numpy(), axis=0)

        return real_lst,fake_lst
    
    def scale_images(self,images, new_shape=(299,299,3)):
        images_list = list()
        
        for image in images:
            new_image = resize(image, new_shape, 0)
            images_list.append(new_image)
        return asarray(images_list)
 

    def calculate_inception_score(self, n_split=10, eps=1E-16):
        inception = InceptionV3()
        scores = list()
        images = self.fake
        n_part = floor(images.shape[0] / n_split)
        for i in range(n_split):
            ix_start, ix_end = i * n_part, (i+1) * n_part
            subset = images[ix_start:ix_end]
            subset = subset.astype('float32')
            
            subset = self.scale_images(images=subset, new_shape=(299,299,3))
            subset = preprocess_input(subset)
            p_yx = inception.predict(subset)
            p_y = expand_dims(p_yx.mean(axis=0), 0)
            kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
            sum_kl_d = kl_d.sum(axis=1)
            avg_kl_d = mean(sum_kl_d)
            is_score = exp(avg_kl_d)
            scores.append(is_score)
        is_avg, is_std = mean(scores), std(scores)
        return is_avg, is_std

 
    def calculate_fid(self):
        act1 = self.incV3.predict(self.images1)
        act2 = self.incV3.predict(self.images2)
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        ssdiff = numpy.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=16384, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=8, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--config-path', type=str, default='./configs/model.yaml', help='Path to configs.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    args.dataset_path = r"/home/ygong2832/file/ILSVRC/Data/CLS-LOC/test_sub"
    args.checkpoint_path = r"./taming-transformers/logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt"
    args.config_path = r"./taming-transformers/logs/vqgan_imagenet_f16_16384/configs/model.yaml"

    testpic = Test(args)

