#!/usr/bin/env python
# coding: utf-8
# %%
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
# from tensorflow import keras
# from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray


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
# from utils import load_data, plot_images
import sys
sys.path.insert(0, '/home/ygong2832/taming-transformers')
sys.path.insert(0, '/home/ygong2832/taming-transformers/taming')
# import model
class TestTransformer:
    def __init__(self, args):
        self.sos_token = args.sos_token
        
        self.model = self.load_transformer(args).to(device=args.device)
        
        self.score = self.calculate_inception_score(args)

                
   # def scale_images(self,images, new_shape):
    #    images_list = list()
     #   for image in images:
      #      # resize with nearest neighbor interpolation
       #     new_image = resize(image, new_shape, 0)
        #    # store
         #   images_list.append(new_image)
        #return asarray(images_list)
    
    def load_transformer(self, args):
        model = VQGANTransformer(args)
        print(args.transformer_checkpoint_path)
        sd = torch.load(args.transformer_checkpoint_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)   
#         model.load_checkpoint2(path=args.transformer_checkpoint_path)
        model = model.eval()
        return model
    
    def load_testdata(self, args):
        test_data = ImagePaths(args.test_dataset_path, size=256)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        return test_loader
    
#     def test(self, args):
#         test_dataset = load_data(args)
#         with tqdm(range(len(test_dataset))) as pbar:
#             for i, imgs in zip(pbar, test_dataset):
#                 imgs = imgs.to(device=args.device)
#                 for j in range(len(imgs)): #不确定要不要加这个for在batchsize里
#                     log, sampled_imgs = self.model.log_images(imgs[j][None])
#                     vutils.save_image(log["rec"], os.path.join("test", f"{epoch}.jpg"), nrow=4)
                    
        
 

    def calculate_inception_score(self, args, n_split=10, eps=1E-16):
        image = self.load_testdata(args) ## 改arg里数据来源
        image=image.to(device=args.device) #?? 可能不需要
        shuffle(test_dataset)
        
        # enumerate splits of images/predictions
        scores = list()
        n_part = floor(images.shape[0] / n_split)
        for i in range(n_split):
        
            ix_start, ix_end = i * n_part, (i+1) * n_part
            subset = images[ix_start:ix_end]
        
            subset = subset.astype('float32')
            # predict p(y|x)
            
            log, sampled_imgs = self.model.log_images(subset[:][None])
            p_yx = log["rec"] #有可能是2，因为多个图片会多一个维度，rec变为2
            
            # calculate p(y)
            p_y = expand_dims(p_yx.mean(axis=0), 0)
            # calculate KL divergence using log probabilities
            kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
            # sum over classes
            sum_kl_d = kl_d.sum(axis=1)
            # average over images
            avg_kl_d = mean(sum_kl_d)
            # undo the log
            is_score = exp(avg_kl_d)
            # store
            scores.append(is_score)
        # average across images
        is_avg, is_std = mean(scores), std(scores)
        print('score', is_avg, is_std)
        
        return is_avg, is_std
 


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to vqgan checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=10, help='Input batch size for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')
    parser.add_argument('--config-path', type=str, default='./configs/model.yaml', help='Path to configs.')
    parser.add_argument('--transformer-checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to transformer checkpoint.')
    parser.add_argument('--test-dataset-path', type=str, default='./test', help='Path to test data.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    args.dataset_path = r"/home/ygong2832/file/ILSVRC/Data/CLS-LOC/train_sub2"
    args.checkpoint_path = r"./taming-transformers/logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt"
    args.config_path = r"./taming-transformers/logs/vqgan_imagenet_f16_16384/configs/model.yaml"
    args.transformer_checkpoint_path = r"./checkpoints/transformer_epoch_30.pt"
#     r"/home/ygong2832/checkpoints/transformer_epoch_39.pt"
    args.test_dataset_path = r"/home/ygong2832/file/ILSVRC/Data/CLS-LOC/test_sub"

    testpic = TestTransformer(args)


# %%





# %%





# def calculate_inception_score(images, n_split=10, eps=1E-16):
    
#     model = ae
#     # enumerate splits of images/predictions
#     scores = list()
#     n_part = floor(images.shape[0] / n_split)
#     for i in range(n_split):
        
#         ix_start, ix_end = i * n_part, (i+1) * n_part
#         subset = images[ix_start:ix_end]
        
#         ## for i, imgs in zip(pbar, subset):
        
#         subset = subset.astype('float32')
#         # predict p(y|x)
#         p_yx = model.predict(subset)
#         # calculate p(y)
#         p_y = expand_dims(p_yx.mean(axis=0), 0)
#         # calculate KL divergence using log probabilities
#         kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
#         # sum over classes
#         sum_kl_d = kl_d.sum(axis=1)
#         # average over images
#         avg_kl_d = mean(sum_kl_d)
#         # undo the log
#         is_score = exp(avg_kl_d)
#         # store
#         scores.append(is_score)
#     # average across images
#     is_avg, is_std = mean(scores), std(scores)
#     return is_avg, is_std
 


# shuffle(X_test)
# is_avg, is_std = calculate_inception_score(X_test)
# print('score', is_avg, is_std)

