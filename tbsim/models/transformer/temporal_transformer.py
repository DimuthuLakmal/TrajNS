import torch
from torch import nn

from tbsim.models.diffuser_helpers import SinusoidalPosEmb
from tbsim.models.transformer.transformer_encoder import TransformerEncoder
from tbsim.models.transformer.transformer_decoder import TransformerDecoder


class TemporalTransformer(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(TemporalTransformer, self).__init__()

        self.encoder = TransformerEncoder(
            dim_model=encoder_config['dim_model'],
            num_heads=encoder_config['num_heads'],
            num_encoder_layers=encoder_config['num_layers'],
            dropout_p=0.2,
            max_seq_len=encoder_config['max_seq_len']+1,
            map=True,
            agent_hist=False).to('cuda')

        dim_model = encoder_config['dim_model']
        t_dim = 256
        cond_dim = 384
        proj_input_dim = cond_dim + t_dim
        self.fc_proj = nn.Linear(in_features=proj_input_dim, out_features=dim_model)
        self.time_mlp = nn.Sequential(
        #     SinusoidalPosEmb(t_dim),
        #     nn.Linear(t_dim, t_dim * 2),
        #     nn.Mish(),
        #     nn.Linear(t_dim * 2, t_dim),
        # )

        # self.decoder = TransformerDecoder(dim_model=decoder_config['dim_model'],
        #                                   num_heads=decoder_config['num_heads'],
        #                                   num_layers=decoder_config['num_layers']).to('cuda')

    def forward(self, x_noise, aux_info, t):
        # t = self.time_mlp(t).unsqueeze(dim=1)
        x_cond = torch.cat([aux_info['map_global_feat_hist'], t], dim=1)

        enc_out = self.encoder(x_cond)
        dec_out = self.decoder(x_noise, enc_out)

        return dec_out
