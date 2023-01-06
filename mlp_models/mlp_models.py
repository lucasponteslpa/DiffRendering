import torch
import torch.nn as nn

class PositionalEncoding(torch.nn.Module):
    def __init__(self, n_freqs=7,device='cuda'):
        super(PositionalEncoding,self).__init__()

        self.n_freqs = n_freqs
        self.freq_bands = 2.**torch.linspace(0, n_freqs-1, n_freqs, dtype=torch.float32, device=device).unsqueeze(0)

    def forward(self, x):
        x_in = x.unsqueeze(-1)

        mul_res = torch.matmul(x_in, self.freq_bands)
        r_cos = torch.cos(mul_res)
        r_sin = torch.sin(mul_res)

        out = torch.cat((r_cos,r_sin), -1)
        out = out.view(-1,x.shape[-1]*2*self.n_freqs)

        return out



class SimpleColorMLP(torch.nn.Module):
    def __init__(self, pos_enc_len = 7, in_channels=3, n_mlp_layers=3, out_channels=3):
        super(SimpleColorMLP,self).__init__()

        self.pos_enc = PositionalEncoding(pos_enc_len)

        in_feats = in_channels*2*pos_enc_len
        net_width = 256
        early_mlp = []
        for layer_idx in range(n_mlp_layers-1):
            early_mlp.append(nn.Linear(in_feats, net_width))
            early_mlp.append(nn.ReLU())
            in_feats = net_width
        early_mlp.append(nn.Linear(in_feats, out_channels))
        early_mlp.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*early_mlp)

    def forward(self, x):
        x_enc = self.pos_enc(x)
        out_color = self.mlp(x_enc.to(torch.float32))

        return out_color

