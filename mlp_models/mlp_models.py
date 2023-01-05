import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, n_freqs=7):
        super(PositionalEncoding,self).__init__()

        self.n_freqs = n_freqs
        self.freq_bands = 2.**torch.linspace(0, n_freqs-1, n_freqs, dtype=torch.float64).unsqueeze(0)

    def forward(self, x):
        x_in = x.unsqueeze(-1)

        mul_res = torch.matmul(x_in, self.freq_bands)
        r_cos = torch.cos(mul_res)
        r_sin = torch.sin(mul_res)

        out = torch.cat((r_cos,r_sin), -1)
        out = out.view(-1,x.shape[-1]*2*self.n_freqs)

        return out



class SimpleColorMLP(torch.nn.Module):
    def __init__(self, device, pos_enc_len = 7, in_channels=3):
        super(SimpleColorMLP,self).__init__()

        self.pos_enc = PositionalEncoding(pos_enc_len)
        self.layer1 = torch.Linear(in_channels*2*pos_enc_len, 280)