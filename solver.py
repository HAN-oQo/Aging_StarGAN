from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
# from age_classifier_v2 import age_classifier_v2

# CACD_loss = 'CE'
# Temperature = 10.0

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, celeba_loader, rafd_loader, CACD_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.celeba_loader = celeba_loader
        self.rafd_loader = rafd_loader
        self.CACD_loader = CACD_loader
        self.attention = config.attention
        self.inter = config.inter
        self.test_version = config.test_version
        
        self.self_attention_model = config.self_attention_model
        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.lambda_KL = config.lambda_KL
        self.lambda_ID = config.lambda_ID
        self.lambda_ma = config.lambda_ma
        self.lambda_ms = config.lambda_ms
        self.lambda_gan = config.lambda_gan
        self.lambda_inter = config.lambda_inter
        self.lambda_tri = config.lambda_tri



        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs
        self.age_group = config.age_group
        self.age_group_mode = config.age_group_mode
        self.age_estimation = config.age_estimation 
        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.classifier_dir = config.classifier_dir
        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step
        self.loss_type = config.loss_type
        self.temperature = config.temperature

        
        self.identity_loss = nn.L1Loss().to(self.device)
        self.MSELoss = nn.MSELoss().to(self.device)
        self.L1Loss = nn.L1Loss().to(self.device)

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # classifier = age_classifier_v2.Hybridmodel(self.age_group)
        # classifier.load_state_dict(torch.load(self.classifier_dir))
        # self.classifier = classifier.to(self.device)
        
        # print("Classifier loaded")
    def mask_activation_loss(self, x):
        return torch.mean(x)

    def mask_smooth_loss(self, x):
        return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

    def build_model(self):
        """Create a generator and a discriminator."""
        if self.dataset in ['CelebA', 'RaFD', 'CACD']:
            self.G = Generator(self.attention ,self.g_conv_dim, self.c_dim, self.g_repeat_num)
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 
        elif self.dataset in ['Both']:
            self.G = Generator(self.g_conv_dim, self.c_dim+self.c2_dim+2, self.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim+self.c2_dim, self.d_repeat_num)
            
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage), strict = False)
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage), strict = False)

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out



    def create_labels(self, c_org, c_dim=6, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
            
            elif dataset == 'CACD':
                if self.age_group_mode == 2:
                    c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
                else:
                    # print(c_org)
                    # print(c_org.size())
                    c_trg = torch.zeros(c_org.size())
                    # print(c_trg.size())
                    c_trg[:, i] = 1

            c_trg_list.append(c_trg.to(self.device))
        # print(c_org)
        # print(c_trg)
        # print(c_trg.size())
        # print(c_trg_list)
        # print(c_trg_list)
        
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA', loss_type = 'BCE'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)
        elif dataset == 'CACD':
            if loss_type == 'BCE': 
                loss = nn.BCEWithLogitsLoss(size_average = False)
                return loss(logit, target) / logit.size(0)
            elif loss_type == 'LOGIT_MSE': # Trick: prediction with log_softmax, target with softmax
                # Wrong implementation. KLDIV loss input should be a prob distribution. Not hard label.
                loss = nn.KLDivLoss(reduction = 'batchmean')
                prediction = F.log_softmax(logit/self.temperature, dim = 1)
                target_tensor = F.softmax(target/self.temperature , dim = 1) # target = [0 1 0 0]
                return loss(prediction, target_tensor) * (self.temperature**2)
            else:
                # one hot vector to label?
                loss = nn.CrossEntropyLoss()
                return loss(logit, target)
    
    def make_label_usable(self, label):
        if self.age_group_mode == 2:
            label = label.long()
            label = label.view(label.size(0))
        return label
    def GANLoss(self, pred, target= True):
        if target == True:
            real_label = Tensor(pred.size()).fill_(1.0)
            return self.MSELoss(pred, real_label)
        else:
            fake_label = Tensor(pred.size()).fill_(0.0)
            return self.MSELoss(pred, fake_label)

    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        elif self.dataset == 'CACD':
            data_loader = self.CACD_loader

        
       

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        filename, x_fixed, c_org = next(data_iter)
        c_org = self.make_label_usable(c_org)

        print(c_org)
        x_fixed = x_fixed.to(self.device)
    
        if self.dataset == 'CACD':
            c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.age_group)
        else:
            c_fixed_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                filename, x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                filename, x_real, label_org = next(data_iter)
            
            label_org = self.make_label_usable(label_org)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.inter == True:
                rand_idx_A = torch.randperm(label_org.size(0))
                label_trg_A = label_org[rand_idx_A]

            if self.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.dataset == 'RaFD' :
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)
            elif self.dataset =='CACD' and self.age_group_mode == 2 :                
                c_org = self.label2onehot(label_org, self.c_dim)
                c_trg = self.label2onehot(label_trg, self.c_dim)
                if self.inter == True:
                    c_trg_A = self.label2onehot(label_trg_A, self.c_dim)
            elif self.dataset =='CACD' :                
                c_org = label_org.clone()
                c_trg = label_trg.clone()

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

            if self.inter == True:
                c_trg_A = c_trg_A.to(self.device)
                label_trg_A = label_trg_A.to(self.device)

            # self.classifier = self.classifier.to(self.device)
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org, self.dataset, 'CE')

            # Compute loss with fake images.
            if self.attention != True:
                x_fake = self.G(x_real, c_trg)
            else:
                x_fake, mask_fake  = self.G(x_real, c_trg)
                x_fake = mask_fake * x_real + (1-mask_fake) * x_fake
                #######
                # x_id , mask_id = self.G(x_real, c_org)
                # x_id = mask_id * x_real + (1-mask_id) * x_id
                # out_src_id , out_cls_id = self.D(x_id.detach())
                # d_loss_id = torch.mean(out_src_id)
                #######
                if self.inter == True:
                    x_fake_A, mask_fake_A = self.G(x_real, c_trg_A)
                    x_fake_A = mask_fake_A * x_real + (1-mask_fake_A) * x_fake_A    
                    x_fake_A_0, mask_fake_A_0 = self.G(x_fake_A, c_trg)
                    x_fake_A_0 = mask_fake_A_0 * x_fake_A + (1 -mask_fake_A_0) * x_fake_A_0
                    x_fake_0_A, mask_fake_0_A = self.G(x_fake, c_trg_A)
                    x_fake_0_A = mask_fake_0_A * x_fake + (1-mask_fake_0_A) * x_fake_0_A

 
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)
            if self.inter == True:
                out_src_A ,out_cls_A = self.D(x_fake_A.detach())
                d_loss_fake_A = torch.mean(out_src_A)
                # inter relation gan loss
                # ============================================
                out_src_A_0, out_cls_A_0 = self.D(x_fake_A_0.detach())
                d_loss_fake_A_0 = self.GANLoss(out_src_A_0, False)
                out_src_0_A, out_cls_0_A = self.D(x_fake_0_A.detach())
                d_loss_fake_0_A = self.GANLoss(out_src_0_A, False)
                d_loss_inter_gan = d_loss_fake_0_A + d_loss_fake_A_0
                # =============================================
            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)
            
            ####
            # alpha_id = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            # x_hat_id = (alpha_id * x_real.data + (1 - alpha_id) * x_id.data).requires_grad_(True)
            # out_src_id, _ = self.D(x_hat_id)
            # d_loss_gp_id = self.gradient_penalty(out_src_id, x_hat_id)

            # d_loss_fake = d_loss_fake + d_loss_id
            # d_loss_gp = d_loss_gp + d_loss_gp_id
            #####
            if self.inter == True:
                alpha_A = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat_A = (alpha_A * x_real.data + (1 - alpha_A) * x_fake_A.data).requires_grad_(True)
                out_src_A, _ = self.D(x_hat_A)
                d_loss_gp_A = self.gradient_penalty(out_src_A, x_hat_A)

            # Backward and optimize.
            if self.inter != True:
                d_loss = self.lambda_gan * (d_loss_real + d_loss_fake) + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            else:
                d_loss = d_loss_real + d_loss_fake + d_loss_fake_A \
                        + self.lambda_cls * d_loss_cls + self.lambda_gp * (d_loss_gp + d_loss_gp_A) \
                        + self.lambda_gan * (d_loss_inter_gan) 
            # d_loss = d_loss_real + d_loss_fake  + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            if self.inter == True: 
                loss['D/loss_fake_A'] = d_loss_fake_A.item()
                loss['D/loss_gp_A'] = d_loss_gp_A.item()
                loss['D/loss_inter_gan'] = d_loss_inter_gan.item()
                

            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            
            if (i+1) % self.n_critic == 0:

                # Identity mapping
                if self.attention != True:
                    x_id = self.G(x_real, c_org)
                else:
                    x_id, mask_id = self.G(x_real, c_org)
                    x_id = mask_id * x_real + (1-mask_id) * x_id

                out_src_id, out_cls_id = self.D(x_id) 
                # g_loss_id = - torch.mean(out_src_id)
                g_loss_cls_id = self.classification_loss(out_cls_id, label_org, self.dataset, 'CE')
                
                #g_loss_identity = self.identity_loss(x_id , x_real)

                # Original-to-target domain.
                if self.attention != True:
                    x_fake = self.G(x_real, c_trg)
                else:
                    x_fake, mask_fake  = self.G(x_real, c_trg)
                    x_fake = mask_fake * x_real + (1-mask_fake) * x_fake

                out_src, out_cls = self.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg, self.dataset,'CE')
                
                # g_loss_fake = g_loss_fake + g_loss_id
                g_loss_cls = g_loss_cls + g_loss_cls_id
                
                margin_power = torch.abs(label_org - label_trg)
                # print(margin_power, margin_power.size())
                # print(x_real.size())
                # print(x_fake.size())
                # print(torch.mean(torch.abs(x_real - x_id), dim= [1,2,3], keepdim = False), torch.mean(torch.abs(x_real - x_id)).size())

                margin = 0.02 * margin_power
                # print(margin, margin.size())
                #TripleMarginLoss = nn.TripletMarginLoss(margin, p =1).to(self.device)
                TripletMarginLoss = torch.mean(torch.abs(x_real - x_id), dim= [1,2,3], keepdim = False) - torch.mean(torch.abs(x_real-x_fake), dim= [1,2,3], keepdim = False)
                # print(TripletMarginLoss, TripletMarginLoss.size())
                TripletMarginLoss = torch.max ((TripletMarginLoss + margin), torch.Tensor([0.]).to(self.device))
                # print(TripletMarginLoss, TripletMarginLoss.size())
                # g_loss_tri = margin_power * TripletMarginLoss(x_real, x_id, x_fake)
                g_loss_tri = TripletMarginLoss.sum() #/ torch.nonzero(TripletMarginLoss.data).size(0)
                # g_loss_tri = torch.mean(TripletMarginLoss)
                # Target-to-original domain.
                if self.attention != True:
                    x_reconst = self.G(x_fake, c_org)
                else:
                    # trial : x_fake , c_org , x_id, c_trg
                    x_reconst, mask_reconst = self.G(x_id, c_trg)
                    x_reconst = mask_reconst * x_id + (1-mask_reconst) * x_reconst


                    #g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))
                    g_loss_rec = torch.mean(torch.abs(x_fake - x_reconst))

                    # print(mask_fake, mask_fake.size())
                    # print(mask_reconst, mask_reconst.size())

                    g_mask_activation_loss =  self.mask_activation_loss(mask_fake) + self.mask_activation_loss(mask_reconst) + self.mask_activation_loss(mask_id)
                    #g_mask_smooth_loss = self.mask_smooth_loss(mask_fake) + self.mask_smooth_loss(mask_reconst)

                    if self.inter == True:
                        x_fake_A, mask_fake_A = self.G(x_real, c_trg_A)
                        x_fake_A = mask_fake_A * x_real + (1-mask_fake_A) * x_fake_A    
                        x_fake_A_0, mask_fake_A_0 = self.G(x_fake_A, c_trg)
                        x_fake_A_0 = mask_fake_A_0 * x_fake_A + (1-mask_fake_A_0) * x_fake_A_0
                        x_fake_0_A, mask_fake_0_A = self.G(x_fake, c_trg_A)
                        x_fake_0_A = mask_fake_0_A * x_fake + (1-mask_fake_0_A) * x_fake_0_A
                        
                        out_src_A, out_cls_A = self.D(x_fake_A)
                        out_src_A_0, out_cls_A_0 = self.D(x_fake_A_0)
                        out_src_0_A, out_cls_0_A = self.D(x_fake_0_A)

                        g_loss_fake_A = - torch.mean(out_src_A)
                        g_loss_fake_A_0 = self.GANLoss(out_src_A_0, True)
                        g_loss_fake_0_A = self.GANLoss(out_src_0_A, True)

                        g_loss_cls_A = self.classification_loss(out_cls_A, label_trg_A, self.dataset,'CE')
                        g_loss_cls_A_0 = self.classification_loss(out_cls_A_0, label_trg, self.dataset,'CE')
                        g_loss_cls_0_A = self.classification_loss(out_cls_0_A, label_trg_A, self.dataset,'CE')

                        g_mask_activation_loss_A = self.mask_activation_loss(mask_fake_A) + self.mask_activation_loss(mask_fake_A_0) + self.mask_activation_loss(mask_fake_0_A)
                        g_mask_smooth_loss_A = self.mask_smooth_loss(mask_fake_A) + self.mask_smooth_loss(mask_fake_0_A) + self.mask_smooth_loss(mask_fake_A_0)

                        g_mask_activation_loss = g_mask_activation_loss + g_mask_activation_loss_A
                        g_mask_smooth_loss = g_mask_smooth_loss + g_mask_smooth_loss_A
                        g_loss_inter_gan = g_loss_fake_0_A + g_loss_fake_A_0
                        g_loss_cls = g_loss_cls + g_loss_cls_A
                        g_loss_inter_cls = g_loss_cls_A_0 + g_loss_cls_0_A
                        g_loss_inter = self.L1Loss(x_fake_A_0, x_fake) + self.L1Loss(x_fake_0_A, x_fake_A)
                         

                # real_pred, reg_loss0 = self.classifier(x_real)
                # fake_pred, reg_loss1 = self.classifier(x_fake)
                # print(real_pred, real_pred.size())
                # print(fake_pred, fake_pred.size())
                # KLloss =  self.classification_loss( fake_pred, real_pred, self.dataset,'LOGIT_MSE')
                # Backward and optimize.
                if self.attention != True:
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                else:
                    if self.inter != True:
                        g_loss =  self.lambda_gan * g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls + self.lambda_tri * g_loss_tri + self.lambda_ma *g_mask_activation_loss #+ self.lambda_ms * g_mask_smooth_loss 
                    else: 
                        g_loss = g_loss_fake + g_loss_fake_A + g_loss_inter_gan \
                                + self.lambda_rec * g_loss_rec \
                                + self.lambda_cls * (g_loss_cls+g_loss_inter_cls)\
                                + self.lambda_tri * g_loss_tri + self.lambda_inter * g_loss_inter \
                                + self.lambda_ma *g_mask_activation_loss #+ self.lambda_ms * g_mask_smooth_loss 
                # g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_KL * KLloss
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                if self.attention == True:
                    loss['G/loss_tri'] = g_loss_tri.item()
                    loss['G/loss_mask_activation'] = g_mask_activation_loss.item()
                    #loss['G/loss_mask_smooth'] = g_mask_smooth_loss.item()
                    if self.inter == True:
                        loss['G/loss_inter'] = g_loss_inter.item()
                        loss['G/loss_inter_gan'] = g_loss_inter_gan.item()
                        loss['G/loss_inter_cls'] = g_loss_inter_cls.item()

                # loss['G/loss_KL_div'] = KLloss.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                if self.attention != True:
                    with torch.no_grad():
                        x_fake_list = [x_fixed]
                        for c_fixed in c_fixed_list:
                            x_fake_list.append(self.G(x_fixed, c_fixed))
                        x_concat = torch.cat(x_fake_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        print('Saved real and fake images into {}...'.format(sample_path))
                else:
                     with torch.no_grad():
                        x_fake_list = [x_fixed]
                        #x_mask_list = [x_fixed]
                        x_mask_list = []
                        for c_fixed in c_fixed_list:
                            images, masks = self.G(x_fixed, c_fixed)
                            images = masks * x_fixed + (1-masks) * images
                            x_fake_list.append(images)
                            x_mask_list.append(masks)

                        x_concat = torch.cat(x_fake_list, dim=3)
                        mask_concat = torch.cat(x_mask_list, dim=3)
                        sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                        mask_sample_path = os.path.join(self.sample_dir, '{}-masks.jpg'.format(i+1))
                        save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                        save_image(mask_concat.data.cpu(), mask_sample_path, nrow=1, padding=0, normalize = True)
                        print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Train StarGAN with multiple datasets."""        
        # Data iterators.
        celeba_iter = iter(self.celeba_loader)
        rafd_iter = iter(self.rafd_loader)

        # Fetch fixed inputs for debugging.
        filename, x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.device)
        c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_fixed.size(0), self.c_dim).to(self.device)           # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_fixed.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
        mask_celeba = self.label2onehot(torch.zeros(x_fixed.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
        mask_rafd = self.label2onehot(torch.ones(x_fixed.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #
                
                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter
                
                try:
                    filename, x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.celeba_loader)
                        filename, x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.rafd_loader)
                        filename, x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                x_real = x_real.to(self.device)             # Input images.
                c_org = c_org.to(self.device)               # Original domain labels.
                c_trg = c_trg.to(self.device)               # Target domain labels.
                label_org = label_org.to(self.device)       # Labels for computing classification loss.
                label_trg = label_trg.to(self.device)       # Labels for computing classification loss.

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.D(x_real)
                out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(out_cls, label_org, dataset)

                # Compute loss with fake images.
                if self.attention != True:
                    x_fake = self.G(x_real, c_trg)
                else:
                    x_fake, mask_fake  = self.G(x_real, c_trg)
                    x_fake = mask_fake * x_real + (1-mask_fake) * x_fake

                out_src, _ = self.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = self.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()
            
                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.n_critic == 0:
                    if self.attention != True:
                        x_id = self.G(x_real, c_org)
                    else:
                        x_id, mask_id = self.G(x_real, c_org)
                        x_id = mask_id * x_real + (1-mask_id) * x_id 
                    
                    
                    # Original-to-target domain.
                    if self.attention != True:
                        x_fake = self.G(x_real, c_trg)
                    else:
                        x_fake, mask_fake  = self.G(x_real, c_trg)
                        x_fake = mask_fake * x_real + (1-mask_fake) * x_fake
                        
                    out_src, out_cls = self.D(x_fake)
                    out_cls = out_cls[:, :self.c_dim] if dataset == 'CelebA' else out_cls[:, self.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    
                    x_reconst = self.G(x_fake, c_org)
                    
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(et, i+1, self.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat([c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat([zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        elif self.dataset == 'CACD':
            data_loader = self.CACD_loader
        
        with torch.no_grad():
            for i, (filename, x_real, c_org) in enumerate(data_loader):
                if self.test_version == 0:

                    print(c_org)
                    if self.dataset == 'CACD':
                        filename = "".join(filename)
                        for k in range(self.age_group):
                            dir_name = 'age_group{}'.format(k)
                            if not os.path.exists(os.path.join(self.result_dir, dir_name)):
                                os.makedirs(os.path.join(self.result_dir, dir_name))

                    if self.dataset == 'CelebA' or self.dataset == 'RaFD':
                        # Prepare input images and target domain labels.
                        filename = "".join(filename)
                        filenum = filename.split('.')[0]
                        # print(filenum)

                        if not os.path.exists(os.path.join(self.result_dir, 'input')):
                            os.makedirs(os.path.join(self.result_dir, 'input'))

                        if not os.path.exists(os.path.join(self.result_dir, 'output')):
                            os.makedirs(os.path.join(self.result_dir, 'output'))
                        
                        real_dir = os.path.join(self.result_dir, 'input')
                        fake_dir = os.path.join(self.result_dir, 'output')

                        if not os.path.exists(os.path.join(fake_dir, 'aging')):
                            os.makedirs(os.path.join(fake_dir, 'aging'))
                        aging_dir = os.path.join(fake_dir, 'aging')

                        real_path = os.path.join(real_dir, '{}.jpg'.format(filenum))
                        save_image(self.denorm(x_real), real_path)
                        
                    
                        
                    x_real = x_real.to(self.device)
                    if self.dataset == 'CelebA':
                        c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)
                    elif self.dataset == 'CACD':
                        c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, None)

                        # Translate images.

                    x_fake_list = [x_real]
                    for j, c_trg in enumerate(c_trg_list):
                        if self.attention != True:
                            x_fake = self.G(x_real, c_trg)
                        else:
                            x_fake, mask_fake = self.G(x_real, c_trg)
                            x_fake = mask_fake * x_real + (1-mask_fake)* x_fake
                        # x_fake_list.append(self.G(x_real, c_trg))
                        if self.dataset == 'CelebA':
                            if j==0:
                                result_path = os.path.join(fake_dir, 'Black_Hair-{}.jpg'.format(filenum))
                            elif j==1:
                                result_path = os.path.join(fake_dir, 'Blond_Hair-{}.jpg'.format(filenum))
                            
                            elif j==2:
                                result_path = os.path.join(fake_dir, 'Brown_Hair-{}.jpg'.format(filenum))

                            elif j==3:
                                result_path = os.path.join(fake_dir, 'Gender-{}.jpg'.format(filenum))

                            elif j==4:
                                aging_path = os.path.join(aging_dir, 'Aging-{}.jpg'.format(filenum))
                                save_image(self.denorm(x_fake.data.cpu()), aging_path)
                                result_path = os.path.join(fake_dir, 'Aging-{}.jpg'.format(filenum))
                        
                        elif self.dataset == 'CACD':
                            age_path = os.path.join(self.result_dir, 'age_group{}'.format(j))
                            result_path = os.path.join(age_path, 'age{}_{}.jpg'.format(j, i))
                            
                        save_image(self.denorm(x_fake.data.cpu()), result_path)
                    
                    
                    print('Saved real and fake images into result path, filenum: {}...'.format(i))
                else:
                    
                    x_real = x_real.to(self.device)
                    c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                    # Translate images.
                    x_fake_list = [x_real]
                    for c_trg in c_trg_list:
                        x_fake, mask_fake = self.G(x_real, c_trg)
                        x_fake = mask_fake * x_real + (1-mask_fake)* x_fake
                        x_fake_list.append(x_fake)

                    # Save the translated images.
                    x_concat = torch.cat(x_fake_list, dim=3)
                    result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(result_path))


                # Save the translated images.
                
                # x_concat = torch.cat(x_fake_list, dim=3)
                # result_path = os.path.join(self.result_dir, 'translated-{}.jpg'.format(filenum))
                # save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                # print('Saved real and fake images into {}...'.format(result_path))

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        with torch.no_grad():
            for i, (filename, x_real, c_org) in enumerate(self.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_celeba_list = self.create_labels(c_org, self.c_dim, 'CelebA', self.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.c_dim).to(self.device)            # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.c2_dim).to(self.device)             # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))