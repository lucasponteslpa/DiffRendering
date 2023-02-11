import torch
import torch.nn as nn
import numpy as np
import math

class EnsembleLinear(nn.Module):

    def __init__(self, in_features, out_features, ensemble_size, bias=True):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.Tensor(ensemble_size, in_features, out_features)
        if bias:
            self.biases = torch.Tensor(ensemble_size, 1, out_features)
        else:
            self.register_parameter('biases', None)
        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weights:
            w.transpose_(0, 1)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            w.transpose_(0, 1)

        self.weights = nn.parameter.Parameter(self.weights)

        if self.biases is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights[0].T)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.biases, -bound, bound)
            self.biases = nn.parameter.Parameter(self.biases)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.repeat(self.ensemble_size, 1, 1)
        return torch.baddbmm(self.biases, input, self.weights)

    def extra_repr(self) -> str:
        return 'ensemble_size = {}, in_features={}, out_features={}, biases={}'.format(
            self.ensemble_size, self.in_features, self.out_features, self.biases is not None
        )

class EnsembleLinearBlock(nn.Module):

    def __init__(self, layers_shape, activation=nn.LeakyReLU(), bias=True):
        super().__init__()

        self.weights_layers = []
        self.biases_layers = [] if bias else None
        self.activation = activation
        self.ensemble_size = layers_shape[0][0]
        for l in layers_shape:
            ensemble_size = l[0]
            in_features = l[1]
            out_features = l[2]

            weights = torch.Tensor(ensemble_size, in_features, out_features)
            self.weights_layers.append(weights)

            if bias:
                biases = torch.Tensor(ensemble_size, 1, out_features)
                self.biases_layers.append(biases)

        self.reset_parameters()

    def reset_parameters(self):
        for l, weights in enumerate(self.weights_layers):
            for w in weights:
                w.transpose_(0, 1)
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                w.transpose_(0, 1)

            self.weights_layers = nn.parameter.Parameter(weights)

            if self.biases_layers is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights[0].T)
                bound = 1 / nn.math.sqrt(fan_in)
                nn.init.uniform_(self.biases, -bound, bound)
                self.biases_layers[l] = nn.parameter.Parameter(self.biases_layers[l])

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.repeat(self.ensemble_size, 1, 1)
        for w, b in zip(self.weights_layers, self.biases_layers):
            out_hid = torch.baddbmm(b, x, w)
            # x =
        return torch.baddbmm(self.biases, input, self.weights)

    def extra_repr(self) -> str:
        return 'ensemble_size = {}, in_features={}, out_features={}, biases={}'.format(
            self.ensemble_size, self.in_features, self.out_features, self.biases is not None
        )

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

class ExtendedSimpleColorMLP(torch.nn.Module):
    def __init__(self,
                 feature_in=9,
                 pose_len=72,
                 pos_enc_len = 7,
                 in_channels=2,
                 n_mlp_layers=8,
                 n_layers_coords=3,
                 n_layers_feat_coords=3,
                 out_channels=3,
                 out_coords=9,
                 in_feat=9):
        super(ExtendedSimpleColorMLP,self).__init__()

        self.pos_enc = PositionalEncoding(pos_enc_len)

        in_enc_feats = (in_channels*2*pos_enc_len)
        self.in_feats = (in_channels*2*pos_enc_len)
        net_width = 256
        early_mlp = []
        for layer_idx in range(n_layers_coords-1):
            early_mlp.append(nn.Linear(in_enc_feats, net_width))
            early_mlp.append(nn.LeakyReLU())
            in_enc_feats = net_width
        early_mlp.append(nn.Linear(in_enc_feats, out_coords))
        early_mlp.append(nn.LeakyReLU())

        self.early_mlp = nn.Sequential(*early_mlp)

        in_feats = in_feat
        self.in_feats = in_enc_feats + in_feat
        net_width = 256
        early_mlp = []
        for layer_idx in range(n_mlp_layers-1):
            early_mlp.append(nn.Linear(in_feats, net_width))
            early_mlp.append(nn.LeakyReLU())
            in_feats = net_width
        early_mlp.append(nn.Linear(in_feats, out_channels))
        early_mlp.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*early_mlp)

    def forward(self, x, features):
        x_enc = self.pos_enc(x)
        # breakpoint()
        x_coords = self.early_mlp(x_enc)
        # mlp_in = torch.cat((x_coords,features), dim=1)
        mlp_in = (x_coords*features)
        # mlp_in = torch.cat((mlp_in,pose), dim=1)
        out_color = self.mlp(mlp_in.to(torch.float32))

        # out_color = self.mlp(features.to(torch.float32))

        return out_color

class NeuralTextureSMPL(torch.nn.Module):
    def __init__(self, tex_shape, input_len=72, enc_len=9, n_channels=3, n_mlp_layers=3, div_factor=2, device='cuda'):
        super(NeuralTextureSMPL,self).__init__()

        self.device = device
        self.tex_shape = tex_shape
        self.input_len = input_len
        self.enc_len = enc_len
        self.n_channels = n_channels
        self.n_mlp_layers = n_mlp_layers
        self.div_factor = div_factor

        # in_feats = input_len
        # net_width = 1024
        # early_mlp = []
        # for layer_idx in range(n_mlp_layers-1):
        #     early_mlp.append(nn.Linear(in_feats, net_width))
        #     early_mlp.append(nn.ReLU())
        #     in_feats = net_width
        # early_mlp.append(nn.Linear(in_feats, enc_len))
        # early_mlp.append(nn.ReLU())

        # self.enc_mlp = nn.Sequential(*early_mlp)

        self.global_weights_shapes = [( (tex_shape[0]* tex_shape[1]*tex_shape[2])//(div_factor**2), input_len, 32),
                                      ((tex_shape[0]*tex_shape[1]*tex_shape[2])//(div_factor**2), 32, 32),
                                        ((tex_shape[0]*tex_shape[1]*tex_shape[2])//(div_factor**2), 32, 3)]

        # for s in self.global_weights_shapes:
        #     init_params = np.random.uniform(-1.0, 1.0, size=s)
        #     self.global_mtx_params.append(init_params)
        # self.global_mtx_params = torch.tensor(self.global_mtx_params, dtype=torch.float32, device=device, requires_grad=True)

        in_feats = input_len
        net_width = 32
        local_mlp = []
        ensemble_size = (tex_shape[0]* tex_shape[1]*tex_shape[2])//(div_factor**2)
        for layer_idx in range(3):
            local_mlp.append(EnsembleLinear(in_feats, net_width, ensemble_size))
            local_mlp.append(nn.LeakyReLU())
            in_feats = net_width
        local_mlp.append(EnsembleLinear(in_feats, 3, ensemble_size))
        local_mlp.append(nn.Sigmoid())

        self.local_mlp = nn.Sequential(*local_mlp)

        mat_shape_1 = tex_shape[1]
        mat_shape_2 = tex_shape[2]
        coords = torch.linspace(-1.0,1.0,mat_shape_1,dtype=torch.float32, device='cuda')
        x_grid, y_grid = torch.meshgrid(coords, coords, indexing='ij')
        x_coords = x_grid.reshape(-1).unsqueeze(-1)
        y_coords = y_grid.reshape(-1).unsqueeze(-1)
        self.xy_coords = torch.cat((x_coords,y_coords), -1)
        # self.global_mlp = ExtendedSimpleColorMLP(feature_in=enc_len,n_mlp_layers=3)

        # enc_params_shape = (tex_shape[0], tex_shape[1], tex_shape[2], n_channels)
        enc_params_shape = (tex_shape[0], tex_shape[1], tex_shape[2], n_channels, enc_len)
        enc_params_shape = (tex_shape[0], tex_shape[1], tex_shape[2], enc_len)
        # enc_params_shape = (tex_shape[0], tex_shape[1], tex_shape[2], n_channels,input_len)
        # enc_params_shape0 = (tex_shape[0], tex_shape[1], tex_shape[2], input_len,input_len)
        # enc_params_shape1 = (tex_shape[0], tex_shape[1], tex_shape[2], n_channels,input_len)
        init_enc_params = np.random.uniform(-1.0, 1.0, size=enc_params_shape)
        # init_enc_params0 = np.random.uniform(-1.0, 1.0, size=enc_params_shape0)
        # init_enc_params1 = np.random.uniform(-1.0, 1.0, size=enc_params_shape1)
        # self.tex_enc_params = torch.tensor(init_enc_params, dtype=torch.float32, device=device, requires_grad=True)
        # self.tex_enc_params = torch.full(enc_params_shape, 0.2, device='cuda', requires_grad=True)
        # self.tex_enc_params0 = torch.tensor(init_enc_params0, dtype=torch.float32, device=device, requires_grad=True)
        # self.tex_enc_params1 = torch.tensor(init_enc_params1, dtype=torch.float32, device=device, requires_grad=True)
        self.relu =nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        # self.tex_global_params = torch.full(tex_shape, 0.2, device=device, requires_grad=True)

    def forward(self,pose):
        # pose_enc = self.enc_mlp(pose)

        # pose_ext = torch.matmul(torch.ones(self.tex_shape[0], self.tex_shape[1], self.tex_shape[2],1).to(self.device), pose)
        # pose_ext = pose_ext.view(-1,self.input_len)
        # enc_mat_mul = self.tex_enc_params.view(-1, self.n_channels)
        # enc_mat_mul = torch.matmul(self.tex_enc_params, pose_enc.permute(1,0)).squeeze(-1)
        # enc_mat_mul0 = torch.matmul(self.tex_enc_params0, pose.permute(1,0)).squeeze(-1)
        # enc_mat_mul1 = torch.matmul(self.tex_enc_params1, enc_mat_mul0.permute(3,0,1,2)).squeeze(-1)
        # out_inf = self.sigmoid(enc_mat_mul)

        # pose_enc_pad = torch.cat((torch.ones_like(pose_enc), pose_enc), dim=-1).squeeze(0)
        # tex_params_pad = torch.cat((self.tex_enc_params, torch.ones_like(self.tex_enc_params)), dim=-1)
        # pose_enc_tex_tensor = tex_params_pad*pose_enc_pad
        # pose_tex_tensor = torch.ones(((self.tex_shape[0] *self.tex_shape[1] * self.tex_shape[2])//(self.div_factor**2), self.input_len)).to(self.device)*pose.squeeze(0)
        # pose_tex_tensor = pose_tex_tensor.view(-1, self.input_len)
        # hidden_out = torch.einsum('ij, ijk -> ik',pose_tex_tensor,self.global_mtx_params[0])
        # hidden_out = self.relu(hidden_out)
        # hidden_out2 = torch.einsum('ij, ijk -> ik',hidden_out,self.global_mtx_params[1])
        # hidden_out2 = self.relu(hidden_out2)
        # hidden_out3 = torch.einsum('ij, ijk -> ik',hidden_out2,self.global_mtx_params[2])
        # out_feat = self.sigmoid(hidden_out3)
        # out_feat = out_feat.view(self.tex_shape[0], self.tex_shape[1]//self.div_factor, self.tex_shape[2]//self.div_factor, self.enc_len)
        out_feat = self.local_mlp(pose)
        out_inf = out_feat.view(self.tex_shape[0], self.tex_shape[1]//self.div_factor, self.tex_shape[2]//self.div_factor, 3)
        # out_feat = out_feat.permute(0,3,1,2)
        # out_feat_int = torch.nn.functional.interpolate(out_feat,scale_factor=self.div_factor, mode='bicubic')
        # out_inf = out_feat_int.permute(0,2,3,1)
        # for m  in self.global_mtx_params[1:]:
        #     hidden_out = torch.matmul(hidden_out,m)


        # global_mlp_in_feat = out_feat_int.view(-1, self.enc_len)
        # out_inf = self.global_mlp(self.xy_coords,global_mlp_in_feat)
        # out_inf = out_inf.view(1,self.tex_shape[1], self.tex_shape[2],3)
        # tex_sum = self.tex_global_params+enc_inf

        return out_inf