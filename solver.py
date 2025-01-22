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
        self.lr_decay = self.lr / float(self.epochs * (self.n_samples // self.mini_batch_size))
        self.current_step = 0
        
    def step(self):
        self.lr = max(0., self.lr - self.lr_decay)
        for param_groups in self.optimizer.param_groups:
            param_groups['lr'] = self.lr
        self.current_step += 1
        
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'lr': self.lr,
        }

    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.lr = state_dict['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
            
'''
 PyTorch Lightning 用 solver
'''
class LitGAN(pl.LightningModule):
    def __init__(self, config:dict) -> None:        
        super().__init__()
        self.config = config

        self.automatic_optimization = False

        # input_shape=(mel_dim, num_frames), residual_in_channels
        self.generator_noisy2clean = Generator(input_shape=config['input_shape'], 
                                               residual_in_channels=config['residual_in_channels']).to(self.device)
        self.generator_clean2noisy = Generator(input_shape=config['input_shape'],
                                               residual_in_channels=config['residual_in_channels']).to(self.device)
        self.discriminator_noisy = Discriminator(input_shape=config['input_shape'],
                                                 residual_in_channels=config['residual_in_channels']).to(self.device)
        self.discriminator_clean = Discriminator(input_shape=config['input_shape'],
                                                 residual_in_channels=config['residual_in_channels']).to(self.device)

        # 2-step adversarial loss
        self.discriminator_noisy2 = Discriminator(input_shape=config['input_shape'],
                                                  residual_in_channels=config['residual_in_channels']).to(self.device)
        self.discriminator_clean2 = Discriminator(input_shape=config['input_shape'],
                                                  residual_in_channels=config['residual_in_channels']).to(self.device)

        # clean/noisy classifier
        self.clean_noisy_classifier = AuxDiscriminator(input_shape=config['input_shape'],
                                                  residual_in_channels=config['residual_in_channels'], 
                                                  output_class=2)
        self.lambda_clean_noisy = config['lambda_clean_noisy']
        self.cycle_loss_lambda = config['cycle_loss_lambda']
        self.identity_loss_lambda = config['identity_loss_lambda']
        
    def forward(self, data:Tensor, mask:Tensor) -> Tensor:
        return self.generator_noisy2clean(data, mask) 

    def toggle_training_mode(self, generator=True):
        if generator is True:
            self.generator_noisy2clean.train()
            self.generator_clean2noisy.train()
            self.discriminator_noisy.eval()
            self.discriminator_clean.eval()
            self.discriminator_noisy2.eval()
            self.discriminator_clean2.eval()
            self.speaker_classifier.train()
            self.clean_noisy_classifier.train()
        else:
            self.generator_noisy2clean.eval()
            self.generator_clean2noisy.eval()
            self.discriminator_noisy.train()
            self.discriminator_clean.train()
            self.discriminator_noisy2.train()
            self.discriminator_clean2.train()
            self.speaker_classifier.eval()
            self.clean_noisy_classifier.eval()

    def compute_discriminator_loss(self, d_real_noisy, d_fake_noisy, d_cycled_noisy, d_real_noisy2, 
                                   d_real_clean, d_fake_clean, d_cycled_clean, d_real_clean2, valid=False):
        d = {}
        d_loss_noisy_real = torch.mean((1 - d_real_noisy) ** 2)
        d_loss_noisy_fake = torch.mean((1 - d_fake_noisy) ** 2)
        d_loss_noisy = (d_loss_noisy_real + d_loss_noisy_fake)/2.0
        if valid is True:
            d['valid_discriminator_loss_noisy_real'] = d_loss_noisy_real
            d['valid_discriminator_loss_noisy_fake'] = d_loss_noisy_fake
            d['valid_discriminator_loss_noisy'] = d_loss_noisy
        else: 
            d['discriminator_loss_noisy_real'] = d_loss_noisy_real
            d['discriminator_loss_noisy_fake'] = d_loss_noisy_fake
            d['discriminator_loss_noisy'] = d_loss_noisy

        d_loss_clean_real = torch.mean((1 - d_real_clean) ** 2)
        d_loss_clean_fake = torch.mean((1 - d_fake_clean) ** 2)
        d_loss_clean = (d_loss_clean_real + d_loss_clean_fake)/2.0
        if valid is True:
            d['valid_discriminator_loss_clean_real'] = d_loss_clean_real
            d['valid_discriminator_loss_clean_fake'] = d_loss_clean_fake
            d['valid_discriminator_loss_clean'] = d_loss_clean
        else:
            d['discriminator_loss_clean_real'] = d_loss_clean_real
            d['discriminator_loss_clean_fake'] = d_loss_clean_fake
            d['discriminator_loss_clean'] = d_loss_clean

        d_loss_noisy_cycled = torch.mean((0 - d_cycled_noisy) ** 2)
        d_loss_clean_cycled = torch.mean((0 - d_cycled_clean) ** 2)
        d_loss_noisy2_real = torch.mean((1 - d_real_noisy2) ** 2)
        d_loss_clean2_real = torch.mean((1 - d_real_clean2) ** 2)
        d_loss_noisy_2nd = (d_loss_noisy2_real + d_loss_noisy_cycled)/2.0
        d_loss_clean_2nd = (d_loss_clean2_real + d_loss_clean_cycled)/2.0
        if valid is True:
            d['valid_discriminator_loss_noisy_cycled'] = d_loss_noisy_cycled
            d['valid_discriminator_loss_clean_cycled'] = d_loss_clean_cycled
            d['valid_discriminator_loss_noisy2_real'] = d_loss_noisy2_real
            d['valid_discriminator_loss_clean2_real'] = d_loss_clean2_real
            d['valid_discriminator_loss_noisy_2nd'] = d_loss_noisy_2nd
            d['valid_discriminator_loss_clean_2nd'] = d_loss_clean_2nd
        else:
            d['discriminator_loss_noisy_cycled'] = d_loss_noisy_cycled
            d['discriminator_loss_clean_cycled'] = d_loss_clean_cycled
            d['discriminator_loss_noisy2_real'] = d_loss_noisy2_real
            d['discriminator_loss_clean2_real'] = d_loss_clean2_real
            d['discriminator_loss_noisy_2nd'] = d_loss_noisy_2nd
            d['discriminator_loss_clean_2nd'] = d_loss_clean_2nd

        _dsc_loss = (d_loss_noisy + d_loss_clean)/2.0 + (d_loss_noisy_2nd + d_loss_clean_2nd)/2.0
        if valid is True:
            d['valid_discriminator_loss'] = _dsc_loss
        else:
            d['discriminator_loss'] = _dsc_loss

        self.log_dict(d)

        return _dsc_loss
    
    def compute_generator_loss(self, real_noisy, real_clean, cycle_noisy, cycle_clean, 
                               ident_noisy, ident_clean, d_fake_noisy, d_fake_clean,
                               d_fake_cycle_noisy, d_fake_cycle_clean, valid=False):
        d = {}

        # Cycle loss
        _cycle_loss = torch.mean(torch.abs(real_noisy - cycle_noisy)) + torch.mean(torch.abs(real_clean - cycle_clean))
        if valid is True:
            d['valid_cycle_loss'] = _cycle_loss
        else:
            d['cycle_loss'] = _cycle_loss
        # Identity loss
        _ident_loss = torch.mean(torch.abs(real_noisy - ident_noisy)) + torch.mean(torch.abs(real_clean - ident_clean))
        if valid is True:
            d['valid_identity_loss'] = _ident_loss
        else:
            d['identity_loss'] = _ident_loss

        # Generator adversarila loss
        _gen_loss_noisy2clean = torch.mean((1 - d_fake_clean) ** 2)
        _gen_loss_clean2noisy = torch.mean((1 - d_fake_noisy) ** 2)
        if valid is True:
            d['valid_generator_loss_noisy2clean'] = _gen_loss_noisy2clean
            d['valid_generator_loss_clean2noisy'] = _gen_loss_clean2noisy
        else:
            d['generator_loss_noisy2clean'] = _gen_loss_noisy2clean
            d['generator_loss_clean2noisy'] = _gen_loss_clean2noisy

        # Generator 2-step adversarial loss
        _gen_loss_noisy2clean_2nd = torch.mean((1 - d_fake_cycle_clean) ** 2)
        _gen_loss_clean2noisy_2nd = torch.mean((1 - d_fake_cycle_noisy) ** 2)
        if valid is True:
            d['valid_generator_loss_noisy2clean_2nd'] = _gen_loss_noisy2clean_2nd
            d['valid_generator_loss_clean2noisy_2nd'] = _gen_loss_clean2noisy_2nd
        else:
            d['generator_loss_noisy2clean_2nd'] = _gen_loss_noisy2clean_2nd
            d['generator_loss_clean2noisy_2nd'] = _gen_loss_clean2noisy_2nd

        _gen_loss = _gen_loss_noisy2clean + _gen_loss_clean2noisy + \
                    _gen_loss_noisy2clean_2nd + _gen_loss_clean2noisy_2nd + \
                    self.cycle_loss_lambda * _cycle_loss + \
                    self.identity_loss_lambda * _ident_loss
        if valid is True:        
            d['valid_generator_loss'] = _gen_loss
        else:
            d['generator_loss'] = _gen_loss
        self.log_dict(d)

        return _gen_loss

    def compute_clean_noisy_logits_loss(self, real_noisy, fake_noisy, cycle_noisy, ident_noisy,
                                  real_clean, fake_clean, cycle_clean, ident_clean, valid=False):
        d = {}
        logits_real_noisy = self.clean_noisy_classifier(real_noisy)
        logits_fake_noisy = self.clean_noisy_classifier(fake_noisy)
        logits_cycle_noisy = self.clean_noisy_classifier(cycle_noisy)
        logits_ident_noisy = self.clean_noisy_classifier(ident_noisy)
        loss_real_noisy = F.cross_entropy(logits_real_noisy, torch.ones(real_noisy.shape[0], type=torch.int64, device=real_noisy.device))
        loss_fake_noisy = F.cross_entropy(logits_fake_noisy, torch.ones(real_noisy.shape[0], type=torch.int64, device=real_noisy.device))
        loss_cycle_noisy = F.cross_entropy(logits_cycle_noisy, torch.ones(real_noisy.shape[0], type=torch.int64, device=real_noisy.device))
        loss_ident_noisy = F.cross_entropy(logits_ident_noisy, torch.ones(real_noisy.shape[0], type=torch.int64, device=real_noisy.device))
        noisy_loss = loss_real_noisy + loss_fake_noisy + loss_cycle_noisy + loss_ident_noisy
        if valid is True:
            d['valid_noisy_loss'] = noisy_loss
        else:
            d['noisy_loss'] = noisy_loss

        logits_real_clean = self.clean_noisy_classifier(real_clean)
        logits_fake_clean = self.clean_noisy_classifier(fake_clean)
        logits_cycle_clean = self.clean_noisy_classifier(cycle_clean)
        logits_ident_clean = self.clean_noisy_classifier(ident_clean)
        loss_real_clean = F.cross_entropy(logits_real_clean, torch.zeros(real_clean.shape[0], type=torch.int64, device=real_clean.device))
        loss_fake_clean = F.cross_entropy(logits_fake_clean, torch.zeros(real_clean.shape[0], type=torch.int64, device=real_clean.device))
        loss_cycle_clean = F.cross_entropy(logits_cycle_clean, torch.zeros(real_clean.shape[0], type=torch.int64, device=real_clean.device))
        loss_ident_clean = F.cross_entropy(logits_ident_clean, torch.zeros(real_clean.shape[0], type=torch.int64, device=real_clean.device)))
        clean_loss = loss_real_clean + loss_fake_clean + loss_cycle_clean + loss_ident_clean
        if valid is True:
            d['valid_clean_loss'] = clean_loss
        else:
            d['clean_loss'] = clean_loss

        _loss = noisy_loss + clean_loss
        if valid is True:
            d['valid_clean_noisy_loss'] = _loss
        else:
            d['clean_noisy_loss'] = _loss
        self.log_dict(d)

        return _loss
        
    def training_step(self, batch, batch_idx:int) -> Tensor:
        _loss = 0.
        real_clean, mask_clean, real_noisy, mask_noisy = batch

        opt_g, opt_d = self.optimizers()

        # Generator step
        self.toggle_optimizer(opt_g)
        self.toggle_training_mode(generator=True)

        fake_clean = self.forward(real_noisy, mask_noisy)
        cycle_noisy = self.generator_clean2noisy(fake_clean, torch.ones_like(fake_clean))
        fake_noisy = self.generator_clean2noisy(real_clean, mask_clean)
        cycle_clean = self.generator_noisy2clean(fake_noisy, torch.ones_like(fake_noisy))
        ident_noisy = self.generator_clean2noisy(real_noisy, torch.ones_like(real_noisy))
        ident_clean = self.generator_noisy2clean(real_clean, torch.ones_like(real_clean))

        '''
            data: real_noisy, real_clean, fake_noisy, fake_clean, cycle_noisy, cycle_clean

            1) discriminator_noisy     real_noisy=true,  fake_noisy=clean2noisy(clean)=false
            2) discriminator_noisy2    real_noisy=true,  cycle_noisy=clean2noisy(noisy2clean(noisy))=false
            3) discriminator_clean     real_clean=true,  fake_clean=noisy2clean(noisy)=false
            4) discriminator_clean2    real_clean=true,  cycle_noisy=noisy2clean(clean2noisy(clean))=false
        '''

        d_fake_noisy = self.discriminator_noisy(fake_noisy)
        d_fake_clean = self.discriminator_clean(fake_clean)
        
        d_fake_cycle_noisy = self.discriminator_noisy2(cycle_noisy)
        d_fake_cycle_clean = self.discriminator_clean2(cycle_clean)

        '''
            clean/noisy loss
            data: real_noisy, real_clean, fake_noisy, fake_clean, cycle_noisy, cycle_clean
            classified to noisy: real_noisy, cycle_noisy, fake_noisy, ident_noisy
            classified to clean: real_clean, fake_clean, cycle_clean, ident_clean
        '''
        clean_noisy_loss = self.compute_clean_noisy_logits_loss(real_noisy, fake_noisy, cycle_noisy, ident_noisy,
                                                    real_clean, fake_clean, cycle_clean, ident_clean, valid=False)

        gen_loss = self.compute_generator_loss(
            real_noisy, real_clean, cycle_noisy, cycle_clean, 
            ident_noisy, ident_clean, d_fake_noisy, d_fake_clean,
            d_fake_cycle_noisy, d_fake_cycle_clean, valid=False
        )
        gen_loss += self.lambda_clean_noisy * clean_noisy_loss
        self.manual_backward(gen_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Discriminator
        self.toggle_training_mode(generator=False)
        self.toggle_optimizer(opt_d)

        d_real_noisy = self.discriminator_noisy(real_noisy)
        d_real_clean = self.discriminator_clean(real_clean)
        d_real_noisy2 = self.discriminator_noisy2(real_noisy)
        d_real_clean2 = self.discriminator_clean2(real_clean)
        generated_noisy = self.generator_clean2noisy(real_clean, mask_clean)
        d_fake_noisy = self.discriminator_noisy(generated_noisy)
        
        # for 2-step adversarial loss noisy->clean
        cycled_clean = self.generator_noisy2clean(generated_noisy, torch.ones_like(generated_noisy))
        d_cycled_clean = self.discriminator_clean2(cycled_clean)

        generated_clean = self.generator_noisy2clean(real_noisy, mask_noisy)
        d_fake_clean = self.discriminator_clean(generated_clean)
        # for 2-step adversarial loss clean->noisy
        cycled_noisy = self.generator_clean2noisy(generated_clean, torch.ones_like(generated_clean))
        d_cycled_noisy = self.discriminator_noisy2(cycled_noisy)

        dsc_loss = self.compute_discriminator_loss(d_real_noisy, d_fake_noisy, d_cycled_noisy, d_real_noisy2,
                                                   d_real_clean, d_fake_clean, d_cycled_clean, d_real_clean2, valid=False)
        self.manual_backward(dsc_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        #update scheduler using self.current_epoch
        for sch in self.lr_schedulers():
            sch.step()

        # return nothing becuase of manual updates   
        return 

    def validation_step(self, batch, batch_idx: int):
        """
        Validation step for evaluating the Generator and Discriminator.
    
        Args:
            batch: Validation batch containing input data.
            batch_idx: Index of the current batch.
    
        Returns:
            Dictionary containing validation losses for logging.
        """
        real_clean, mask_clean, real_noisy, mask_noisy = batch

        # Set models to evaluation mode
        self.generator_noisy2clean.eval()
        self.generator_clean2noisy.eval()
        self.discriminator_noisy.eval()
        self.discriminator_clean.eval()
        self.discriminator_noisy2.eval()
        self.discriminator_clean2.eval()
        self.clean_noisy_classifier.eval()

        # Forward pass for Generator
        with torch.no_grad():  # Disable gradient computation for validation
            fake_clean = self.forward(real_noisy, mask_noisy)
            cycle_noisy = self.generator_clean2noisy(fake_clean, torch.ones_like(fake_clean))
            fake_noisy = self.generator_clean2noisy(real_clean, mask_clean)
            cycle_clean = self.generator_noisy2clean(fake_noisy, torch.ones_like(fake_noisy))
            ident_noisy = self.generator_clean2noisy(real_noisy, torch.ones_like(real_noisy))
            ident_clean = self.generator_noisy2clean(real_clean, torch.ones_like(real_clean))

            # Compute adversarial losses for Generator
            d_fake_noisy = self.discriminator_noisy(fake_noisy)
            d_fake_clean = self.discriminator_clean(fake_clean)
            d_fake_cycle_noisy = self.discriminator_noisy2(cycle_noisy)
            d_fake_cycle_clean = self.discriminator_clean2(cycle_clean)

            gen_loss = self.compute_generator_loss(
                real_noisy, real_clean, cycle_noisy, cycle_clean,
                ident_noisy, ident_clean, d_fake_noisy, d_fake_clean,
                d_fake_cycle_noisy, d_fake_cycle_clean, valid=True
            )

            clean_noisy_loss = self.compute_clean_noisy_logits_loss(
                real_noisy, fake_noisy, cycle_noisy, ident_noisy,
                real_clean, fake_clean, cycle_clean, ident_clean, valid=True
            )

            gen_loss += self.lambda_clean_noisy * clean_noisy_loss

            # Compute losses for Discriminator
            d_real_noisy = self.discriminator_noisy(real_noisy)
            d_real_clean = self.discriminator_clean(real_clean)
            d_real_noisy2 = self.discriminator_noisy2(real_noisy)
            d_real_clean2 = self.discriminator_clean2(real_clean)
            generated_noisy = self.generator_clean2noisy(real_clean, mask_clean)
            d_fake_noisy = self.discriminator_noisy(generated_noisy)
            cycled_clean = self.generator_noisy2clean(generated_noisy, torch.ones_like(generated_noisy))
            d_cycled_clean = self.discriminator_clean2(cycled_clean)
            generated_clean = self.generator_noisy2clean(real_noisy, mask_noisy)
            d_fake_clean = self.discriminator_clean(generated_clean)
            cycled_noisy = self.generator_clean2noisy(generated_clean, torch.ones_like(generated_clean))
            d_cycled_noisy = self.discriminator_noisy2(cycled_noisy)

            dsc_loss = self.compute_discriminator_loss(
                d_real_noisy, d_fake_noisy, d_cycled_noisy, d_real_noisy2,
                d_real_clean, d_fake_clean, d_cycled_clean, d_real_clean2, valid=True
            )

        # Return losses for further analysis
        return {
            "valid_gen_loss": gen_loss.item(),
            "valid_speaker_loss": speaker_loss.item(),
            "valid_clean_noisy_loss": clean_noisy_loss.item(),
            "valid_dsc_loss": dsc_loss.item()
        }


    def configure_optimizers(self):
        gen_optimizer = torch.optim.Adam(list(self.generator_noisy2clean.parameters())
                                         +list(self.generator_clean2noisy.parameters()),
                                        **self.config['gen_optimizer'])
        gen_scheduler = CustomLRScheduler(gen_optimizer, **self.config['gen_scheduler'])

        dsc_optimizer = torch.optim.Adam(list(self.discriminator_noisy.parameters())
                                         +list(self.discriminator_clean.parameters())
                                         +list(self.discriminator_noisy2.parameters())
                                         +list(self.discriminator_clean2.parameters()),
                                        **self.config['dsc_optimizer'])
        dsc_scheduler = CustomLRScheduler(dsc_optimizer, **self.config['dsc_scheduler'])

        return [gen_optimizer, dsc_optimizer], [gen_scheduler, dsc_scheduler]
