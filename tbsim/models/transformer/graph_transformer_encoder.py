import torch
from torch import nn

from tbsim.models.gcn.gcn_fixed_w import GCNConv_Fixed_W
from tbsim.models.transformer.encoder_block import EncoderBlock
from tbsim.models.transformer.positional_encoding import PositionalEncoder


class GraphTransformerEncoder(nn.Module):
    def __init__(self,
            dim_model,
            num_heads,
            num_encoder_layers,
            dropout_p,
            max_seq_len):
        super(GraphTransformerEncoder, self).__init__()

        self.dropout_p = dropout_p
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_encoder_layers
        out_dim = dim_model

        # embedding and positional encoder
        self.hist_emb = nn.Linear(3, dim_model)
        self.positional_encoder = PositionalEncoder(max_seq_len, dim_model, matrix_dim=4)

        # encoder attention blocks
        self.layers = nn.ModuleList(
            [EncoderBlock(
                embed_dim=dim_model,
                num_heads=self.num_heads,
                src_dropout=.2,
                ff_dropout=0.2,
                expansion_factor=4
            ) for i in range(self.num_layers)])

        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(in_channels=dim_model, out_channels=dim_model, kernel_size=3, stride=1, padding=1)
             for _ in range(self.num_layers)])

        self.fc_out = nn.Linear(dim_model, out_dim)

        self.gcn = GCNConv_Fixed_W(
            in_channels=dim_model,
            out_channels=dim_model,
            improved=False,
            cached=False,
            normalize=False,
            add_self_loops=False
        )

        self.batch_norm = nn.BatchNorm2d(1)

        self.lin_graph = nn.Linear(1, dim_model)

    def _get_edge_index(self, n_nodes):
        return [[x for x in range(n_nodes)], [0] * n_nodes]

    def forward(self, x):
        edge_weights = x['edge_weight']
        edge_idx = x['edge_index']
        x = x['all_hist_feat']
        x = self.hist_emb(x)
        out_e = self.positional_encoder(x)

        for enc_layer, conv_layer in zip(self.layers, self.conv_layers):
            # out_e = conv_layer(out_e.transpose(1, 2)).transpose(1, 2)
            q, k, v = out_e, out_e, out_e
            out_e = enc_layer(q, k, v)  # output of temporal encoder layer

        graph_w = out_e.permute(0, 2, 1, 3)
        graph_x = x.permute(0, 2, 1, 3)
        b, t, n, f = graph_x.shape
        graph_x = graph_x.reshape(b, t * n, f)
        graph_w = graph_w.reshape(b, t * n, f)

        graph_out = []
        for w_batch, x_batch, weight_batch, edge_idx_batch in zip(graph_w, graph_x, edge_weights, edge_idx):
            x_batch_out = self.gcn(w_batch, x_batch, edge_idx_batch, weight_batch)
            graph_out.append(x_batch_out.reshape(t, n, -1))

        graph_out = torch.stack(graph_out).permute(0, 3, 2, 1)  # B, F, N, T
        graph_out = self.batch_norm(graph_out).permute(0, 2, 3, 1)

        return self.lin_graph(graph_out)