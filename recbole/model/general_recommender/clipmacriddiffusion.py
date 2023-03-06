# -*- coding: utf-8 -*-
# @Time   : 2020/12/23
# @Author : Yihong Guo
# @Email  : gyihong@hotmail.com

# UPDATE
# @Time    : 2021/6/30,
# @Author  : Xingyu Pan
# @email   : xy_pan@foxmail.com

r"""
MacridVAE
################################################
Reference:
    Jianxin Ma et al. "Learning Disentangled Representations for Recommendation." in NeurIPS 2019.
Reference code:
    https://jianxinma.github.io/disentangle-recsys.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType

import math


class ClipMacridDiffusion(GeneralRecommender):
    r"""MacridVAE is an item-based collaborative filtering model that learns disentangled representations from user
    behavior and simultaneously ranks all items for each user.
    We implement the model following the original author.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(ClipMacridDiffusion, self).__init__(config, dataset)

        self.layers = config["encoder_hidden_size"]
        self.embedding_size = config["embedding_size"]
        self.drop_out = config["dropout_prob"]
        self.kfac = config["kfac"]
        self.tau = config["tau"]
        self.nogb = config["nogb"]
        self.anneal_cap = config["anneal_cap"]
        self.total_anneal_steps = config["total_anneal_steps"]
        self.regs = config["reg_weights"]
        self.std = config["std"]
        #####
        self.n_steps = config['n_steps']
        self.batch_size = config['train_batch_size']
        self.layers = config["mlp_hidden_size"]
        self.lat_dim = config["latent_dimension"]

        # Pre-calculate different terms for closed form

        self.betas = torch.linspace(config['min_beta'], config['max_beta'], config['n_steps']).to(
            self.device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)


        self.update = 0

        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)
        self.encode_layer_dims = (
            [self.n_items] + self.layers + [self.embedding_size] # * 2
        )

        self.encoder = self.mlp_layers(self.encode_layer_dims)

        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        pretrained_item_emb = dataset.get_preload_weight('iid')
        self.mm_representation = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_item_emb), freeze=False).type(torch.FloatTensor)
        
        
        self.k_embedding = nn.Embedding(self.kfac, self.embedding_size)

        self.l2_loss = EmbLoss()
        # parameters initialization

        # Time embedding
        self.time_emb_dim = self.embedding_size + self.mm_representation.weight.shape[-1]
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
                nn.ReLU()
            )
        # parameters initialization

        self.apply(xavier_normal_initialization)

    # Return forward diffusion sample that is equal for every k.
    def forward_diffusion_sample(self, rating_matrix, t):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        
        cores = F.normalize(self.k_embedding.weight, dim=1)
        items = F.normalize(self.item_embedding.weight, dim=1)

        rating_matrix = F.normalize(rating_matrix)
        rating_matrix = F.dropout(rating_matrix, self.drop_out, training=self.training)

        cates_logits = torch.matmul(items, cores.transpose(0, 1)) / self.tau

        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        else:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode

        meanlist = []
        varlist = []
        for k in range(self.kfac):
            cates_k = cates[:, k].reshape(1, -1)
            # encoder
            x_k = rating_matrix * cates_k

            X = self.encoder(x_k)
            
            noise = torch.randn_like(X)
            sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, X.shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                self.sqrt_one_minus_alphas_cumprod, t, X.shape
            )
            # mean + variance
            mean, var =  sqrt_alphas_cumprod_t.to(self.device) * X.to(self.device) \
            + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)

            meanlist += mean,
            varlist += var,

        return meanlist, varlist


    # Return a forward diffusion sample for each user prototype, k.
    def forward_diffusion_sample_k(self, rating_matrix, t):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        
        cores = F.normalize(self.k_embedding.weight, dim=1)
        items = F.normalize(self.item_embedding.weight, dim=1)

        rating_matrix = F.normalize(rating_matrix)
        rating_matrix = F.dropout(rating_matrix, self.drop_out, training=self.training)

        cates_logits = torch.matmul(items, cores.transpose(0, 1)) / self.tau

        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        else:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode

        meanlist = []
        varlist = []
        for k in range(self.kfac):
            cates_k = cates[:, k].reshape(1, -1)
            # encoder
            x_k = rating_matrix * cates_k

            X = self.encoder(x_k)
            
            noise = torch.randn_like(X)
            sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, X.shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                self.sqrt_one_minus_alphas_cumprod, t, X.shape
            )
            # mean + variance
            mean, var =  sqrt_alphas_cumprod_t.to(self.device) * X.to(self.device) \
            + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)

            meanlist += mean,
            varlist += var,

        return meanlist, varlist

    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = (
            torch.arange(user.shape[0])
            .to(self.device)
            .repeat_interleave(self.history_item_id.shape[1], dim=0)
        )
        rating_matrix = (
            torch.zeros(1).to(self.device).repeat(user.shape[0], self.n_items)
        )
        rating_matrix.index_put_(
            (row_indices, col_indices), self.history_item_value[user].flatten()
        )
        return rating_matrix
    
    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu()-1)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=self.std)
            return mu + epsilon * std
        else:
            return mu
        
    def diffusion(self, x, t,):

        # Account for conditioning vector (MM Information)
        #x = torch.concat((x, conditioning), dim=1)

        # Obtain constance values
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)   

        # Get model output
        time_emb = self.time_mlp(t)     
        
        model_output = x + time_emb

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)

        noise = torch.randn_like(x)

        z = model_mean + torch.sqrt(posterior_variance_t) * noise 

        return z, model_mean, posterior_variance_t

    def forward(self, rating_matrix, t, conditioning):

        cores = F.normalize(self.k_embedding.weight, dim=1)
        items = F.normalize(self.item_embedding.weight, dim=1)

        rating_matrix = F.normalize(rating_matrix)
        rating_matrix = F.dropout(rating_matrix, self.drop_out, training=self.training)

        cates_logits = torch.matmul(items, cores.transpose(0, 1)) / self.tau

        if self.nogb:
            cates = torch.softmax(cates_logits, dim=-1)
        else:
            cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=-1)
            cates_mode = torch.softmax(cates_logits, dim=-1)
            cates = self.training * cates_sample + (1 - self.training) * cates_mode

        probs = None
        mulist = []
        logvarlist = []
        noisepredlist = []
        conditionlist = []
        for k in range(self.kfac):
            cates_k = cates[:, k].reshape(1, -1)
            # encoder
            x_k = rating_matrix * cates_k
            h = self.encoder(x_k)

            h_ = torch.concat((h, conditioning), dim=1)

            # Diffusion takes place of commented out lines below from MultiVAE architecture.
            z, mu, logvar = self.diffusion(h_, t)

            noise_pred, condition = z[:, :items.shape[1]], z[:, items.shape[1]:]

            noisepredlist += noise_pred,
            mulist += mu, #.append(mu)
            logvarlist += logvar, #.append(logvar)
            conditionlist += condition,

            # decoder
            z_k = F.normalize(noise_pred, dim=1)
            logits_k = torch.matmul(z_k, items.transpose(0, 1)) / self.tau
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k
            probs = probs_k if (probs is None) else (probs + probs_k)

        logits = torch.log(probs)

        return logits, noisepredlist, mulist, logvarlist, conditionlist

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        rating_matrix = self.get_rating_matrix(user)

        mm_info = F.normalize(self.mm_representation.weight, dim=1)

        user_mm_info = torch.Tensor([(rating * mm_info.T).detach().numpy() for rating in rating_matrix]).type(torch.float32)
        conditioning = torch.amax(user_mm_info, axis=2)
        user_mm_info_avg = torch.mean(user_mm_info, dim=2)

        # t is uniform random variable from {1,2,...,n_steps}
        t = torch.randint(1, self.n_steps+1, (rating_matrix.shape[0],), device=self.device).long()

        # anneal calcualtion - need to look into!
        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        # Forward process with h and random t
        z, noise, mu, logvar, conditionlist = self.forward(rating_matrix, t, conditioning)

        # KL Loss
        #kl_loss = None
        #for i in range(self.kfac):
        #    kl_ = -0.5 * torch.mean(torch.sum(1 + logvar[i] - logvar[i].exp(), dim=1))
        #    kl_loss = kl_ if (kl_loss is None) else (kl_loss + kl_)

        # CE Loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        # Diffusion Loss + Condition Loss
        x_noisylist, noiselist = self.forward_diffusion_sample(rating_matrix, t)
        diffusion_loss = 0
        condition_loss = 0
        for i in range(self.kfac):
            noise_pred, _, _ = self.diffusion(x_noisylist[i], t)
            dl_ = -(F.mse_loss(noiselist[i], noise_pred))
            diffusion_loss = dl_ if (diffusion_loss is None) else (diffusion_loss + dl_)

            cl_ = -(F.l1_loss(conditionlist[i], conditioning))
            condition_loss = cl_ if (condition_loss is None) else (condition_loss + cl_)

        
        #if self.regs[0] != 0 or self.regs[1] != 0:
        #    return ce_loss + kl_loss * anneal + self.reg_loss()

        return ce_loss + diffusion_loss + condition_loss #+ kl_loss * anneal 

    def reg_loss(self):
        r"""Calculate the L2 normalization loss of model parameters.
        Including embedding matrices and weight matrices of model.
        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_1, reg_2 = self.regs[:2]
        loss_1 = reg_1 * self.item_embedding.weight.norm(2)
        loss_2 = reg_1 * self.k_embedding.weight.norm(2)
        loss_3 = 0
        for name, parm in self.encoder.named_parameters():
            if name.endswith("weight"):
                loss_3 = loss_3 + reg_2 * parm.norm(2)
        return loss_1 + loss_2 + loss_3

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        rating_matrix = self.get_rating_matrix(user)

        mm_info = F.normalize(self.mm_representation.weight, dim=1)

        user_mm_info = torch.Tensor([(rating * mm_info.T).detach().numpy() for rating in rating_matrix]).type(torch.float32)
        conditioning = torch.amax(user_mm_info, axis=2)
        user_mm_info_avg = torch.mean(user_mm_info, dim=2)

        # t = T for evaluation
        t = torch.full((rating_matrix.shape[0],), self.n_steps, device=self.device).long()

        scores, _, _, _, _ = self.forward(rating_matrix, t, conditioning)

        return scores[[torch.arange(len(item)).to(self.device), item]]

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]

        rating_matrix = self.get_rating_matrix(user)

        mm_info = F.normalize(self.mm_representation.weight, dim=1)

        user_mm_info = torch.Tensor([(rating * mm_info.T).detach().numpy() for rating in rating_matrix]).type(torch.float32)
        conditioning = torch.amax(user_mm_info, axis=2)
        user_mm_info_avg = torch.mean(user_mm_info, dim=2)

        # t = T for evaluation
        t = torch.full((rating_matrix.shape[0],), self.n_steps, device=self.device).long()

        scores, _, _, _, _ = self.forward(rating_matrix, t, conditioning)

        return scores.view(-1)
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings