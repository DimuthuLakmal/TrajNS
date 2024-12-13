import torch
from torch import nn
from torch.cuda.amp import autocast
from transformers import RobertaModel, DebertaModel

from peft import LoraConfig, get_peft_model

# from tbsim.models.deberta.deberta import DebertaWithSingleOutput
from tbsim.models.diffuser_helpers import SinusoidalPosEmb
from tbsim.models.transformer.graph_transformer_encoder import GraphTransformerEncoder
from tbsim.models.transformer.transformer_encoder import TransformerEncoder


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(SpatioTemporalTransformer, self).__init__()

        # self.image_encoder = TransformerEncoder(
        #     dim_model=encoder_config['dim_model'],
        #     num_heads=encoder_config['num_heads'],
        #     num_encoder_layers=encoder_config['num_layers'],
        #     dropout_p=0.2,
        #     max_seq_len=encoder_config['max_seq_len']+1,
        #     map=True,
        #     agent_hist=False).to('cuda')

        # self.llm_data_encoder = DebertaWithSingleOutput().to('cuda')
        self.deberta_model = DebertaModel.from_pretrained("microsoft/deberta-base")
        # Configure LoRA
        lora_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",  # Task type: Sequence Classification
            inference_mode=False,  # Not in inference mode; for training
            r=8,  # Rank of LoRA matrices
            lora_alpha=16,  # Scaling factor for LoRA
            lora_dropout=0.1,  # Dropout rate
        )
        # Wrap the RoBERTa model with LoRA
        self.llm_data_encoder = get_peft_model(self.deberta_model, lora_config)
        self.llm_proj = nn.Linear(self.deberta_model.config.hidden_size, 64)

        self.graph_encoder = GraphTransformerEncoder(
            dim_model=64,
            num_heads=encoder_config['num_heads'],
            num_encoder_layers=encoder_config['num_layers'],
            dropout_p=0.2,
            max_seq_len=encoder_config['max_seq_len']).to('cuda')


        dim_model = encoder_config['dim_model']
        # t_dim = 256
        cond_dim = 384
        # proj_input_dim = cond_dim + t_dim
        self.fc_enc_proj = nn.Linear(in_features=cond_dim, out_features=dim_model)
        # self.time_mlp = nn.Sequential(
        #     SinusoidalPosEmb(t_dim),
        #     nn.Linear(t_dim, t_dim * 2),
        #     nn.Mish(),
        #     nn.Linear(t_dim * 2, t_dim),
        # )

        # self.decoder = TransformerDecoder(dim_model=decoder_config['dim_model'],
        #                                   num_heads=decoder_config['num_heads'],
        #                                   num_layers=decoder_config['num_layers']).to('cuda')

    def forward(self, aux_info):
        with autocast():
            llm_enc_out = self.llm_data_encoder(aux_info['llm_input_ids'], aux_info['llm_attention_mask'])
            llm_enc_out = self.llm_proj(llm_enc_out.last_hidden_state[:, 0, :])

        graph_enc_out = self.graph_encoder(aux_info)
        # image_enc_out = self.image_encoder(aux_info['map_global_feat_hist'])

        enc_out = torch.cat([graph_enc_out, aux_info['map_global_feat_hist'], llm_enc_out], dim=-1)

        # dec_out = self.decoder(x_noise, enc_out)
        # enc_out = enc_out.reshape((enc_out.shape[0], enc_out.shape[1] * enc_out.shape[2]))
        enc_out = self.fc_enc_proj(enc_out)

        return enc_out

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
