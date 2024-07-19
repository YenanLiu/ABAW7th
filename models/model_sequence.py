import torch
import torch.nn as nn
import models.models_vit as models_vit

from pytorch_tcn import TCN

#################################      MTLTransformer        ###########################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MTLTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dims):
        super(MTLTransformer, self).__init__()
        
        self.input_linear = nn.Linear(input_dim, hidden_dims[0])  # 先转换输入维度
        self.pos_encoder = PositionalEncoding(hidden_dims[0])
        self.layers = nn.ModuleList()
        num_layers = len(hidden_dims)
        for i in range(num_layers):
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dims[i], nhead=num_heads[i])
            self.layers.append(nn.TransformerEncoder(encoder_layer, num_layers=1))
            if i < num_layers - 1:
                self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))  # 转换隐藏维度
        
    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                x = layer(x)
        return x

#################################      LSTM        ###########################################
class MultiLayerLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(MultiLayerLSTM, self).__init__()
        
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(
                nn.LSTM(input_dim, hidden_dim, batch_first=True)
            )
            input_dim = hidden_dim  # 下一层的输入维度是上一层的隐藏维度
            
    def forward(self, x):
        for layer in self.layers:
            x, _ = layer(x)
        return x
        
class MTLTemporalModel(nn.Module):
    def __init__(self, config):
        super(MTLTemporalModel, self).__init__()
        self.time_model = config["time_model"]
        fea_amount = sum([config["AU"], config["EXPR"], config["VA_Arousal"], config["VA_Valence"]])
        self.input_dim = config["input_dim"] * fea_amount

        if self.time_model == "Transformer":
            self.tramsformer = MTLTransformer(self.input_dim, config["num_heads"], config["t_hidden_dims"])
            last_dim = config["t_hidden_dims"][-1]
        if self.time_model == "LSTM":
            self.lstm = MultiLayerLSTM(self.input_dim, config["hidden_dim"])
            last_dim = config["hidden_dim"][-1]

        if self.time_model == "TCN":
            self.tcn = TCN(self.input_dim, num_channels=config["num_channels"], kernel_size=config["kernel_size"], dropout=config["dropout"])
            last_dim = config["num_channels"][-1]

        # head
        if config["AU"]:
            self.AUhead = nn.Sequential(
                nn.Linear(last_dim, 12))
            self._initialize_weights(self.AUhead)
        else:
            self.AUhead = nn.Identity()
        
        if config["EXPR"]:
            self.ExprHead = nn.Sequential(
                nn.Linear(last_dim, 8))
            self._initialize_weights(self.ExprHead)
        else:
            self.ExprHead = nn.Identity()
        
        if config["VA"]:
            self.vhead = nn.Sequential(nn.Linear(last_dim, 1))
            self.ahead = nn.Sequential(nn.Linear(last_dim, 1))
            self._initialize_weights(self.vhead)
            self._initialize_weights(self.ahead)
        else:
            self.vhead = nn.Identity()
            self.ahead = nn.Identity()
        
        if config["VA_Arousal"]:
            self.ahead = nn.Sequential(nn.Linear(768, 1))
            self._initialize_weights(self.ahead)
        else:
            self.ahead = nn.Identity()
        
        if config["VA_Valence"]:
            self.vhead = nn.Sequential(nn.Linear(768, 1))
            self._initialize_weights(self.vhead)
        else:
            self.vhead = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LSTM):
                for name, param in layer.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
            elif isinstance(layer, nn.TransformerEncoderLayer):
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        if param.dim() >= 2:
                            nn.init.xavier_uniform_(param)
                        else:
                            nn.init.uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        fea_list = []

        if config["AU"]:
            fea_list.append(x["AU"])
        if config["EXPR"]:
            fea_list.append(x["EXPR"])
        if config["VA_Arousal"]:
            fea_list.append(x["VA_Arousal"])
        if config["VA_Valence"]:
            fea_list.append(x["VA_Valence"])
        
        x = torch.cat(fea_list, dim=-1)

        if self.time_model == "Transformer":
            x = x.permute(1, 0, 2) # (seq_len, batch_size, input_dim)
            x = self.transformer_encoder(x) # [batchsize, N, feature_dim]
            x = x.permute(1, 0, 2)  # [batchsize, N, feature_dim]
        
        if self.time_model == "LSTM":
            x = self.lstm(x)

        if self.time_model == "TCN":
            x = x.transpose(1, 2) # tcn expected the input dimension is [batchsize, channels, sequence_length]
            x = self.tcn(x) # [batchsize, N, num_channels[-1]]

        # For AU
        au_out = self.AUhead(x)

        # For EXPR
        expr_out = self.ExprHead(x)

        # For VA
        vout = self.vhead(x)
        vpred = torch.tanh(vout)
        
        aout = self.ahead(x)
        apred = torch.tanh(aout)
        
        return au_out, expr_out, vpred, apred
 