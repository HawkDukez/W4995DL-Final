import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils_audio import load_data, plot_images
# from sample_visualization import tensor_to_plt
from IPython import display


class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()

        self.train(args)

    def train(self, args):
        train_dataset = load_data(args)
        for epoch in range(args.epochs):
            print("Epoch -", epoch)
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    self.optim.zero_grad()
#                     imgs = imgs.to(device=args.device)
                    imgs['image'] = imgs['input'].to(device=args.device)
                    x = self.model.vqgan.get_input(self.model.vqgan.first_stage_key,imgs)
#                     print(x.shape)
                    
                    logits, target = self.model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                    loss.backward()
                    self.optim.step()
                    pbar.set_postfix(Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4))
                    pbar.update(0)
            shape = x.shape
            log, sampled_imgs = self.model.log_images(x[0,].reshape([1,shape[1],shape[2],shape[3]]))
            vutils.save_image(sampled_imgs, os.path.join("results", f"{epoch}.jpg"), nrow=4)
            plot_images(log)
#             display.display(tensor_to_plt(sampled_imgs, flip_dims=(2,)))
            torch.save(self.model.state_dict(), os.path.join("checkpoints", f"transformer_epoch_{epoch}.pt"))

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    no_decay.add(fpn)

                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add('pos_emb')

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=1e-4, betas=(0.9, 0.95))
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
#     parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
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
#     parser.add_argument('--config-path', type=str, default='./configs/model.yaml', help='Path to configs.')
    parser.add_argument('--logs-path', type=str, default='./logs', help='Path to logs directory.')
    parser.add_argument('--model-name', type=str, default='2021-06-20T16-24-38_vas_transformer', help='Pretrained Model for SpecVQGAN.')
    parser.add_argument('--transformer-checkpoint-path', type=str, default=None, help='Path to transformer checkpoint.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    #args.dataset_path = r"/home/ygong2832/file/ILSVRC/Data/CLS-LOC/train_sub2"
    args.dataset_path = r"/home/ygong2832/audio/SpecVQGAN/data/vas/baby/melspec_10s_22050hz"
#     args.transformer_checkpoint_path =  r"./checkpoints/transformer_epoch_39.pt"
#     args.dataset_path = r"/home/ygong2832/audio/SpecVQGAN/data/vas/small"
#     args.checkpoint_path = r"./taming-transformers/logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt"
#     args.config_path = r"./taming-transformers/logs/vqgan_imagenet_f16_16384/configs/model.yaml"

    train_transformer = TrainTransformer(args)

# %tb 


