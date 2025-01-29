import math
import torch
import torch.nn as nn


class DecoderBase(nn.Module):

    def __init__(self, frequency_encoding_size: int = 96, latent_size: int = 32,
                 d_model: int = 128, nhead: int = 4, is_critic: bool = False,
                 obs_scale: float = 1000.0, act_scale: float = 1000.0,
                 obs_z_in_init_w: float = 0.0, act_z_in_init_w: float = 0.0,
                 act_out_init_w: float = 3e-3,
                 num_transformer_blocks: int = 3, dim_feedforward: int = 256,
                 dropout: float = 0.0, activation: str = "relu"):

        super(DecoderBase, self).__init__()

        # self.frequency_encoding_size = 768 #(*) for add
        # self.latent_size = 768 #latent_size
        # self.d_model = 128 #d_model
        # self.nhead = 2 #nhead
        # self.obs_scale = 1000.0 #obs_scale
        # self.act_scale = 1000.0 #act_scale
        # self.num_transformer_blocks = 3 #num_transformer_blocks
        # self.is_critic = is_critic
        # self.dim_feedforward = 256 #dim_feedforward
        # self.dropout = 0.0 #dropout
        # self.activation = "relu" #activation
        # self.obs_z_in_init_w = 0.0
        # self.act_z_in_init_w = 0.0
        # self.act_out_init_w = 0.003

        # self.frequency_encoding_size = 96 #(*) for concat
        # self.latent_size = 768 #latent_size
        # self.d_model = 128 #d_model
        # self.nhead = 2 #nhead
        # self.obs_scale = 1000.0 #obs_scale
        # self.act_scale = 1000.0 #act_scale
        # self.num_transformer_blocks = 3 #num_transformer_blocks
        # self.is_critic = is_critic
        # self.dim_feedforward = 256 #dim_feedforward
        # self.dropout = 0.0 #dropout
        # self.activation = "relu" #activation
        # self.obs_z_in_init_w = 0.0
        # self.act_z_in_init_w = 0.0
        # self.act_out_init_w = 0.003

        self.frequency_encoding_size = frequency_encoding_size #(*) for concat
        self.latent_size = latent_size
        self.d_model = d_model
        self.nhead = nhead
        self.obs_scale = obs_scale
        self.act_scale = act_scale
        self.num_transformer_blocks = num_transformer_blocks
        self.is_critic = is_critic
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation #activation
        self.obs_z_in_init_w = 0.0
        self.act_z_in_init_w = 0.0
        self.act_out_init_w = 0.003


        self.obs_z_input_layer = nn.Linear(
            self.latent_size + self.frequency_encoding_size, self.d_model)
        self.act_z_input_layer = nn.Linear(self.latent_size + (
            self.frequency_encoding_size if self.is_critic else 0), self.d_model)

        # self.obs_z_input_layer = nn.Linear(self.latent_size, self.d_model)
        if self.obs_z_in_init_w > 0:
            self.obs_z_input_layer.weight.data \
                .uniform_(-self.obs_z_in_init_w, self.obs_z_in_init_w)


        # self.act_z_input_layer = nn.Linear(self.latent_size, self.d_model)
        if self.act_z_in_init_w > 0:
            self.act_z_input_layer.weight.data \
                .uniform_(-self.act_z_in_init_w, self.act_z_in_init_w)

        self.act_output_layer = nn.Linear(self.d_model + self.latent_size, 1)

        self.act_output_layer.bias.data.fill_(0)
        if self.act_out_init_w > 0:
            self.act_output_layer.weight.data \
                .uniform_(-self.act_out_init_w, self.act_out_init_w)

        self.obs_norm = nn.LayerNorm(self.d_model)
        self.act_norm = nn.LayerNorm(self.d_model)

        self.transformer = nn.Transformer(
            d_model=self.d_model, dim_feedforward=self.dim_feedforward,
            nhead=self.nhead, dropout=self.dropout, activation=self.activation,
            num_encoder_layers=self.num_transformer_blocks,
            num_decoder_layers=self.num_transformer_blocks, batch_first=True)

    @staticmethod
    def frequency_encoding(x, d_model):

        idx = torch.arange(0, d_model, 2).to(dtype=x.dtype, device=x.device)
        div_term = torch.exp(idx * (-math.log(10000.0) / d_model))

        x = x.unsqueeze(-1)
        while len(div_term.shape) < len(x.shape):
            div_term = div_term.unsqueeze(0)

        return torch.cat([torch.sin(x * div_term),
                          torch.cos(x * div_term)], dim=-1)

    def forward(self, obs_z, act_z, obs, act=None):

        # earlier we were creating the obs vector by repeating the values, now we are using anymorph's method where
        # each value is projected to a vector of length 'freq_encoding size' with the use of frequency encoding
        obs = (2 / math.sqrt(2.0)) * self.frequency_encoding(
            obs * self.obs_scale, self.frequency_encoding_size)

        # instead of concatenating we are adding
        n_obs_z = torch.cat([obs_z, obs.permute(1,0,2)], dim=2)
        # n_obs_z = torch.add(obs_z, obs.permute(1, 0, 2))

        if self.is_critic:

            assert act is not None, \
                "q function must condition on the action"

            act = (2 / math.sqrt(2.0)) * self.frequency_encoding(
                act * self.act_scale, self.frequency_encoding_size)

            n_act_z = torch.cat([act_z, act], dim=2)
            # n_act_z = torch.add(act_z, act.permute(1, 0, 2))

        else:
            n_act_z = act_z

        n_obs_z = self.obs_z_input_layer(n_obs_z)
        n_act_z = self.act_z_input_layer(n_act_z)

        act = self.transformer(self.obs_norm(n_obs_z), self.act_norm(n_act_z))

        return self.act_output_layer(
            torch.cat([act_z, act], dim=2)).squeeze(2)

        # return self.act_output_layer(
        # torch.add([act_z, act], dim=2)).squeeze(2)

