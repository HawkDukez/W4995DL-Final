import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils_audio import load_testdata, plot_images
from sample_visualization import tensor_to_plt, spec_to_audio_to_st
from IPython import display
import IPython.display as display_audio
import soundfile
# from evaluation.metrics.fid import calculate_fid




class TestTransformer:
    def __init__(self, args):
#         self.model = self.load_transformer(args).to(device=args.device)
        self.model = VQGANTransformer(args).to(device=args.device)
        self.model = self.model.eval()
        self.fid_new, self.fid_rec = self.test(args)
        #         self.score = self.calculate_inception_score(args)
    
    def test(self, args):
        test_dataset = load_testdata(args)
        num = len(test_dataset)
        ##??
        fake_lst, real_lst = np.empty([1,80, 848,1]), np.empty([1,80, 848,1])
#         fid_new_list, fid_rec_list = np.empty([0]), np.empty([0])
        for i, imgs in enumerate(test_dataset):
            print(i)
            imgs['image'] = imgs['input'].to(device=args.device)
            x = self.model.vqgan.get_input(self.model.vqgan.first_stage_key,imgs)
            shape = x.shape
            print('img shape:',shape)
            log, sampled_imgs = self.model.log_images(x)   #x[0,].reshape([1,shape[1],shape[2],shape[3]]))
            vutils.save_image(sampled_imgs, os.path.join("test/images", f"{i}.jpg"), ncol=4)
#             vutils.save_image(log['input'], os.path.join("test/images_compare", f"{i}_input.jpg"), ncol=4)
#             vutils.save_image(log['rec'], os.path.join("test/images_compare", f"{i}_rec.jpg"), ncol=4)
#             vutils.save_image(log['half_new'], os.path.join("test/images_compare", f"{i}_.jpg"), ncol=4)
#             vutils.save_image(log['new_sample'], os.path.join("test/images_compare", f"{i}_rec.jpg"), ncol=4)
            self.tensor_to_audio(args, log, i)
            display.display(plot_images(log))
#             display.display(tensor_to_plt(sampled_imgs, flip_dims=(2,)))
#             fid_new = calculate_fid(log["input"],log["new_sample"])
#             fid_rec = calculate_fid(log["input"],log["rec"])
#             fid_new_list = np.append(fid_new_list, fid_new)
#             fid_rec_list = np.append(fid_rec_list, fid_rec)
        
            real_lst = np.append(real_lst, log["input"].cpu().detach().numpy().transpose(0, 2, 3, 1), axis=0)
            fake_lst = np.append(fake_lst, log["rec"].cpu().detach().numpy().transpose(0, 2, 3, 1), axis=0)
        print("list shape: ", real_lst.shape)
        np.save(os.path.join("test", f'save_real{num}'),fake_lst) 
        np.save(os.path.join("test", f'save_fake{num}'),real_lst) 
#         fid_new = np.mean(fid_new_list)
#         fid_rec = np.mean(fid_rec_list)
#         print("fid of transformer (new): ", fid_new, '; fid of vqgan (reconstruct): ', fid_rec)
#         return fid_new, fid_rec
    
    
    
    
#     @staticmethod
    def tensor_to_audio(self, args, images: dict, i):
        for idx in images:
            x = images[idx]
            waves = spec_to_audio_to_st(x, args.test_dataset_path,
                                    self.model.config.data.params.sample_rate, show_griffin_lim=False,
                                    vocoder=self.model.melgan, show_in_st=False)
            save_path = os.path.join("test/audio", f"{i}_{idx}.wav")
            soundfile.write(save_path, waves['vocoder'], self.model.config.data.params.sample_rate, 'PCM_24')
            print(f'The {idx} sample has been saved @ {save_path}')
            
#             display_audio.Audio(save_path)
    
    def load_transformer(args):
        model = VQGANTransformer(args)
#         model = model.load_checkpoint(args.transformer_checkpoint_path)
        model = model.eval()
        return model
    



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
    parser.add_argument('--logs-path', type=str, default='./logs', help='Path to logs directory.')
    parser.add_argument('--model-name', type=str, default='2021-06-20T16-24-38_vas_transformer', help='Pretrained Model for SpecVQGAN.')


    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    args.dataset_path = r"/home/ygong2832/audio/SpecVQGAN/data/vas/baby/melspec_10s_22050hz"
    args.transformer_checkpoint_path = r"./checkpoints/transformer_epoch_39.pt"
    args.test_dataset_path = r"/home/ygong2832/audio/SpecVQGAN/data/vas/small"

    testpic = TestTransformer(args)

