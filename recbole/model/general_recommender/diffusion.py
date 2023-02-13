# -*- coding: utf-8 -*-
# @Time   : 2020/12/14
# @Author : Yihong Guo
# @Email  : gyihong@hotmail.com

r"""
MultiVAE
################################################
Reference:
    Dawen Liang et al. "Variational Autoencoders for Collaborative Filtering." in WWW 2018.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class Diffusion(GeneralRecommender):
    r"""MultiVAE is an item-based collaborative filtering model that simultaneously ranks all items for each user.

    We implement the MultiVAE model with only user dataloader.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(Diffusion, self).__init__(config, dataset)

        self.n_steps = config['n_steps']
        self.batch_size = config['train_batch_size']
        self.layers = config["mlp_hidden_size"]
        self.lat_dim = config["latent_dimension"]
        #self.network = network.to(device)
        
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

        self.anneal_cap = config["anneal_cap"]
        self.total_anneal_steps = config["total_anneal_steps"]

        #self.model = SimpleUnet(config, dataset)

        self.update = 0

        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix()
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

        self.encode_layer_dims = [self.n_items] + self.layers + [self.lat_dim]
        self.decode_layer_dims = [int(self.lat_dim)] + self.encode_layer_dims[::-1][
            1:
        ]

        self.encoder = self.mlp_layers(self.encode_layer_dims)
        self.decoder = self.mlp_layers(self.decode_layer_dims)

        # Time embedding
        self.time_emb_dim = self.lat_dim
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.time_emb_dim),
                nn.Linear(self.time_emb_dim, self.time_emb_dim),
                nn.ReLU()
            )
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())
        return nn.Sequential(*mlp_modules)

    def forward_diffusion_sample(self, X, t):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """

        #self.logger.nfo(f'X_0 Type: {type(x_0)} X_0 Items: {len(x_0)} X_0 Lenght: {x_0}')

        #x_0 = self.encoder(rating_matrix) # x = Rating Matrix
        
        noise = torch.randn_like(X)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, X.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, X.shape
        )
        # mean + variance
        return sqrt_alphas_cumprod_t.to(self.device) * X.to(self.device) \
        + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)


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
    
    
    def diffusion(self, x, t):

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
    
    
    def forward(self, h, t=None):

        if t is None:
            #t = torch.randint(0, self.n_steps, (h.shape[0],), device=self.device).long()
            t = torch.Tensor([self.n_steps]) # n_steps = T
        # Diffusion takes place of commented out lines below from MultiVAE architecture.
        z, mu, logvar = self.diffusion(h, t)

        #mu = h[:, : int(self.lat_dim / 2)]
        #logvar = h[:, int(self.lat_dim / 2) :]

        #z = self.reparameterize(mu, logvar)

        z_decoded = self.decoder(z)

        return z_decoded, z, mu, logvar

    def calculate_loss(self, interaction, t):

        user = interaction[self.USER_ID]
        rating_matrix = self.get_rating_matrix(user)
        h = F.normalize(rating_matrix)
        h = self.encoder(rating_matrix)

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1.0 * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        z, _, mu, logvar = self.forward(h)

        # KL loss
        kl_loss = (
            -0.5
            * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            * anneal
        )

        # CE loss
        ce_loss = -(F.log_softmax(z, 1) * rating_matrix).sum(1).mean()

        # Diffusion Loss
        x_noisy, noise = self.forward_diffusion_sample(h, t)
        _, noise_pred, _, _ = self.forward(x_noisy, t)
        diffusion_loss = F.l1_loss(noise, noise_pred)

        return ce_loss + kl_loss - diffusion_loss

    def predict(self, interaction):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """

        user = interaction[self.USER_ID]
        rating_matrix = self.get_rating_matrix(user) # x = Rating Matrix

        h = F.normalize(rating_matrix)
        #h = F.dropout(h, self.drop_out, training=self.training)
        h = self.encoder(h)

        scores, _, _, _ = self.forward(h)

        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        rating_matrix = self.get_rating_matrix(user)

        h = F.normalize(rating_matrix)
        #h = F.dropout(h, self.drop_out, training=self.training)
        h = self.encoder(h)

        scores, _, _, _ = self.forward(h)

        return scores.view(-1)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv1d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose1d(out_ch, out_ch, 3, 2, 1)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv1d(out_ch, out_ch, 3, 2, 1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t,):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, )]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


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


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, config, dataset):
        super().__init__()
        image_channels = 1
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv1d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv1d(up_channels[-1], 1, out_dim)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        print('PRe Conv0')
        x = self.conv0(x)
        print(f'Conv0 Done {x.shape}')

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)

        return self.output(x)

    