import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple
from einops import rearrange
from model import Generator, Discriminator, AuxDiscriminator
import math

class CustomLRScheduler(object):
    def __init__(self, optimizer, n_samples, lr, epochs, mini_batch_size) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.n_samples = n_samples
        self.lr = lr
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.lr_decay = self.lr / float(self.num_epochs * (self.n_samples // self.mini_batch_size))

    def step(self):
        self.lr = max(0., self.lr - self.lr_decay)
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] = self.lr

'''
 PyTorch Lightning 用 solver
'''
class LitGAN(pl.LightningModule):
    def __init__(self, config:dict) -> None:        
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.config = config

        self.automatic_optimization = False

        self.generator_DF2NH = Generator().to(self.device)
        self.generator_NH2DF = Generator().to(self.device)

        self.discriminator_DF = Discriminator().to(self.device)
        self.discriminator_NH = Discriminator().to(self.device)

        # 2-step adversarial loss
        self.discriminator_DF2 = Discriminator().to(self.device)
        self.discriminator_NH2 = Discriminator().to(self.device)

    def forward(self, data:Tensor, mask:Tensor) -> Tensor:
        return self.generator_DF2NH(data, mask) # est, est_spk, ctc

    def toggle_training_mode(self, generator=True):
        if generator is True:
            self.generator_DF2NH.train()
            self.generator_NH2DF.train()
            self.discriminator_DF.eval()
            self.discriminator_NH.eval()
            self.discriminator_DF2.eval()
            self.discriminator_NH2.eval()
        else:
            self.generator_DF2NH.eval()
            self.generator_NH2DF.eval()
            self.discriminator_DF.train()
            self.discriminator_NH.train()
            self.discriminator_DF2.train()
            self.discriminator_NH2.train()

    def compute_discriminator_loss(self, d_real_df, d_fake_df, d_cycled_df, d_real_df2, 
                                   d_real_nh, d_fake_nh, d_cycled_nh, d_real_nh2):
        d = {}
        d_loss_DF_real = torch.mean((1 - d_real_df) ** 2)
        d_loss_DF_fake = torch.mean((1 - d_fake_df) ** 2)
        d_loss_DF = (d_loss_DF_real + d_loss_DF_fake)/2.0
        d['discriminator_loss_DF_real'] = d_loss_DF_real
        d['discriminator_loss_DF_fake'] = d_loss_DF_fake
        d['discriminator_loss_DF'] = d_loss_DF

        d_loss_NH_real = torch.mean((1 - d_real_nh) ** 2)
        d_loss_NH_fake = torch.mean((1 - d_fake_nh) ** 2)
        d_loss_NH = (d_loss_NH_real + d_loss_NH_fake)/2.0
        d['discriminator_loss_NH_real'] = d_loss_NH_real
        d['discriminator_loss_NH_fake'] = d_loss_NH_fake
        d['discriminator_loss_NH'] = d_loss_NH

        d_loss_DF_cycled = torch.mean((0 - d_cycled_df) ** 2)
        d_loss_NH_cycled = torch.mean((0 - d_cycled_nh) ** 2)
        d_loss_DF2_real = torch.mean((1 - d_real_df2) ** 2)
        d_loss_NH2_real = torch.mean((1 - d_real_nh2) ** 2)
        d_loss_DF_2nd = (d_loss_DF2_real + d_loss_DF_cycled)/2.0
        d_loss_NH_2nd = (d_loss_NH2_real + d_loss_NH_cycled)/2.0
        d['discriminator_loss_df_cycled'] = d_loss_DF_cycled
        d['discriminator_loss_nh_cycled'] = d_loss_NH_cycled
        d['discriminator_loss_df2_real'] = d_loss_DF2_real
        d['discriminator_loss_nh2_real'] = d_loss_NH2_real
        d['discriminator_loss_df_2nd'] = d_loss_DF_2nd
        d['discriminator_loss_nh_2nd'] = d_loss_NH_2nd

        _dsc_loss = (d_loss_DF + d_loss_NH)/2.0 + (d_loss_DF_2nd + d_loss_NH_2nd)/2.0
        d['discriminator_loss'] = _dsc_loss

        self.log_dict(d)

        return _dsc_loss
    
    def compute_generator_loss(self, real_df, real_nh, cycle_df, cycle_nh, 
                               ident_df, ident_nh, d_fake_df, d_fake_nh,
                               d_fake_cycle_df, d_fake_cycle_nh):
        d = {}

        # Cycle loss
        _cycle_loss = torch.mean(torch.abs(real_df - cycle_df)) + torch.mean(torch.abs(real_nh - cycle_nh))
        d['cycle_loss'] = _cycle_loss
        # Identity loss
        _ident_loss = torch.mean(torch.abs(real_df - ident_df)) + torch.mean(torch.abs(real_nh - ident_nh))
        d['identity_loss'] = _ident_loss

        # Generator adversarila loss
        _gen_loss_DF2NH = torch.mean((1 - d_fake_nh) ** 2)
        _gen_loss_NH2DF = torch.mean((1 - d_fake_df) ** 2)
        d['generator_loss_DF2NH'] = _gen_loss_DF2NH
        d['generator_loss_NH2DF'] = _gen_loss_NH2DF

        # Generator 2-step adversarial loss
        _gen_loss_DF2NH_2nd = torch.mean((1 - d_fake_cycle_nh) ** 2)
        _gen_loss_NH2DF_2nd = torch.mean((1 - d_fake_cycle_df) ** 2)
        d['generator_loss_DF2NH_2nd'] = _gen_loss_DF2NH_2nd
        d['generator_loss_NH2DF_2nd'] = _gen_loss_NH2DF_2nd

        _gen_loss = _gen_loss_DF2NH + _gen_loss_NH2DF + \
                    _gen_loss_DF2NH_2nd + _gen_loss_NH2DF_2nd + \
                    self.cycle_loss_lambda * _cycle_loss + \
                    self.identity_loss_lambda * _ident_loss
        
        d['generator_loss'] = _gen_loss
        self.log_dict(d)

        return _gen_loss

    def training_step(self, batch, batch_idx:int) -> Tensor:
        _loss = 0.
        nh_data, nh_mask, nh_spk, df_data, df_mask, df_spk = batch

        opt_g, opt_d = self.optimizers()

        # Generator step
        self.toggle_optimizers(opt_g)
        self.toggle_training_mode(generator=True)

        fake_nh = self.forward(df_data, df_mask)
        cycle_df = self.generator_NH2DF(fake_nh, torch.ones_like(fake_nh))
        fake_df = self.generator_NH2DF(nh_data, nh_mask)
        cycle_nh = self.generator_DF2NH(fake_df, torch.ones_like(fake_df))
        ident_df = self.generator_NH2DF(df_data, torch.ones_like(df_data))
        ident_nh = self.generator_DF2NH(nh_data, torch.ones_like(nh_data))

        '''
            data: real_df, real_nh, fake_df, fake_nh, cycle_df, cycle_nh

            1) discriminator_DF     real_df=true,  fake_df=NH2DF(nh)=false
            2) discriminator_DF2    real_df=true,  cycle_df=NH2DF(DF2NH(df))=false
            3) discriminator_NH     real_nh=true,  fake_nh=DF2NH(df)=false
            4) discriminator_NH2    real_nh=true,  cycle_df=DF2NH(NH2DF(nh))=false
        '''

        d_fake_df = self.discriminator_DF(fake_df)
        d_fake_nh = self.discriminator_NH(fake_nh)
        
        d_fake_cycle_df = self.discriminator_DF2(cycle_df)
        d_fake_cycle_nh = self.discriminator_NH2(cycle_nh)

        gen_loss = self.compute_generator_loss(
            df_data, nh_data, cycle_df, cycle_nh, 
            ident_df, ident_nh, d_fake_df, d_fake_nh,
            d_fake_cycle_df, d_fake_cycle_nh
        )
        self.manual_backward(gen_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Discriminator
        self.toggle_training_mode(generator=False)
        self.toggle_optimizers(opt_d)

        d_real_DF = self.discriminator_DF(df_data)
        d_real_NH = self.discriminator_NH(nh_data)
        d_real_DF2 = self.discriminator_DF2(df_data)
        d_real_NH2 = self.discriminator_NH2(nh_data)
        generated_DF = self.generator_NH2DF(nh_data, nh_mask)
        d_fake_DF = self.discriminator_DF(generated_DF)
        
        # for 2-step adversarial loss DF->NH
        cycled_NH = self.generator_DF2NH(generated_DF, torch.ones_like(generated_DF))
        d_cycled_NH = self.discriminator_NH2(cycled_NH)

        generated_NH = self.generator_DF2NH(df_data, df_mask)
        d_fake_NH = self.discriminator_NH(generated_NH)

        # for 2-step adversarial loss NH->DF
        cycled_DF = self.generator_NH2DF(generated_NH, torch.ones_like(generated_NH))
        d_cycled_DF = self.discriminator_DF2(cycled_DF)

        dsc_loss = self.compute_discriminator_loss(d_real_DF, d_fake_DF, d_cycled_DF, d_real_DF2,
                                                   d_real_NH, d_fake_NH, d_cycled_NH, d_real_NH2)
        self.manual_backward(dsc_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        #update scheduler using self.current_epoch
        for sch in self.schedulers():
            sch.step()

        # return nothing becuase of manual updates   
        return 

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        pass

    def configure_optimizers(self):
        gen_optimizer = torch.optim.Adam(list(self.generator_DF2NH.parameters())
                                         +list(self.generator_NH2DF.parameters()),
                                        **self.config['gen_optimizer'])
        gen_scheduler = CustomLRScheduler(gen_optimizer, **self.config['gen_scheduler'])

        dsc_optimizer = torch.optim.Adam(list(self.discriminator_DF.parameters())
                                         +list(self.discriminator_NH.parameters())
                                         +list(self.discriminator_DF2.parameters())
                                         +list(self.discriminator_NH2.parameters()),
                                        **self.config['dsc_optimizer'])
        dsc_scheduler = CustomLRScheduler(dsc_optimizer, **self.config['dsc_scheduler'])

        return [gen_optimizer, dsc_optimizer], [gen_scheduler, dsc_scheduler]
