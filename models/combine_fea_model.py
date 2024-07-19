import torch
import torch.nn as nn
import models.models_vit as models_vit
from pytorch_tcn import TCN


class TemporalConverge(nn.Module):
    def __init__(self, config):
        super(TemporalConverge, self).__init__()
        self.time_model = config["time_model"]
        self.input_dim = config["input_dim"] 

        if self.time_model == "Transformer":
            self.d_model = config.get("d_model", 768)
            self.num_heads = config.get("num_heads", 4)
            self.num_layers = config.get("num_layers", 2)
            self.dropout = config.get("dropout", 0.3)

            self.pre_linear = nn.Linear(self.input_dim, self.d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads, dropout=self.dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        if self.time_model == "LSTM":
            self.hidden_dim = config["hidden_dim"]
            self.num_layers = config["l_num_layers"]
            self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        
        if self.time_model == "TCN":
            self.tcn = TCN(self.input_dim, num_channels=config["num_channels"], kernel_size=config["kernel_size"], dropout=config["dropout"])
        
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
        if self.time_model == "Transformer":
            x = self.pre_linear(x)
            x = x.permute(1, 0, 2)  # [sequence_length, batch_size, feature_dim]
            x = self.transformer_encoder(x)
            x = x.permute(1, 0, 2)  # [batch_size, sequence_length, feature_dim]
        
        if self.time_model == "LSTM":
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            x, _ = self.lstm(x, (h0, c0))

        if self.time_model == "TCN":
            x = x.transpose(1, 2) # tcn expected the input dimension is [batchsize, channels, sequence_length]
            x = self.tcn(x) # [batchsize, N, num_channels[-1]]
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, feat_dim, config):
        super(FeatureFusionModule, self).__init__()
        self.fusion_fc = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim)
        )
    
    def forward(self, fea, au_fea_tensor=None, expr_fea_tensor=None, v_fea_tensor=None):

        # au_fea_tensor = self.norm_au(au_fea_tensor)
        # expr_fea_tensor = self.norm_expr(expr_fea_tensor)
        # v_fea_tensor = self.norm_v(v_fea_tensor)
        
        combined_fea = fea
        if au_fea_tensor is not None:
            combined_fea += au_fea_tensor
        if expr_fea_tensor is not None:
            combined_fea += expr_fea_tensor
        if v_fea_tensor is not None:
            combined_fea += v_fea_tensor
        
        fused_feature = self.fusion_fc(combined_fea)
        
        return fused_feature
    
class MTLModel(nn.Module):
    def __init__(self, config):
        super(MTLModel, self).__init__()
        self.config = config

        self.model = getattr(models_vit, 'vit_base_patch16')(
        global_pool=True,
        drop_path_rate=0.1,
        img_size=224,
        )
        self.model.head = nn.Identity()

        ##########################    Model Load    ###############################
        if config["pretrain_model"]:
            state_dict = torch.load(config["pretrain_model"], map_location=torch.device('cpu'))["model"]
            self.model.load_state_dict(state_dict, strict=False)
            print("load from pretrain_model")
        
        if config["finetune_model"]:
            state_dict = torch.load(config["finetune_model"], map_location=torch.device('cpu'))
            new_state_dict = {key.replace("module.", "").replace("main.main.", ""): item for key, item in state_dict.items()}
            del state_dict
            self.model.load_state_dict(new_state_dict, strict=False)
            print("load from finetune_model")
        
        ############################  Confuse Feature  ###########################
        if config["AU_Fea"] or config["EXPR_Fea"] or config["V_Fea"]:
            self.feat_fusion = FeatureFusionModule(768, config)
        else:
            self.feat_fusion = nn.Identity()

        ##########################    Timporal Model Load    ###############################
        if config["time_model"]:
            self.temporal_convergence = TemporalConverge(config)
            self.active = nn.LeakyReLU(0.1) 
        else:
            self.temporal_convergence = nn.Identity()
            self.active = nn.Identity()

        ##########################    Prediction Head    ###############################
        if config["AU"]:
            self.AUhead = nn.Sequential(
                nn.Linear(768, 12))
            self._initialize_weights(self.AUhead)
        else:
            self.AUhead = nn.Identity()
        
        if config["EXPR"]:
            self.ExprHead = nn.Sequential(
                nn.Linear(768, 8))
            self._initialize_weights(self.ExprHead)
        else:
            self.ExprHead = nn.Identity()

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

    def _initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, inp, au_fea_tensor, expr_fea_tensor, v_fea_tensor, fea_ext = False):  
        if len(inp.shape) > 4:
            bz, seq, c, H, W = inp.shape

            inp = inp.view(bz*seq, c, H, W)
            _, fea = self.model(inp, True)

            fea = fea.view(bz, seq, -1)

            if fea_ext:
                fea = fea.reshape(bz * seq, -1)
                return fea

            # fea add fusion
            if au_fea_tensor is not None or expr_fea_tensor is not None or v_fea_tensor is not None:
                fea = self.feat_fusion(fea, au_fea_tensor, expr_fea_tensor, v_fea_tensor)

            fea = self.active(fea)
            fea = self.temporal_convergence(fea) # [batchsize, N, feature_dim] torch.Size([4, 12, 768])
            
            fea = fea.reshape(bz * seq, -1)
        else:
            _, fea = self.model(inp, True)

            if fea_ext:
                return fea

            if au_fea_tensor is not None or expr_fea_tensor is not None or v_fea_tensor is not None:
                fea = self.feat_fusion(fea, au_fea_tensor, expr_fea_tensor, v_fea_tensor)
            fea = self.active(fea)
            
        if fea_ext:
            return fea
 
        # For AU
        au_out = self.AUhead(fea)

        # For EXPR
        expr_out = self.ExprHead(fea)

        # For VA
        vout = self.vhead(fea)
        vpred = torch.tanh(vout)
        
        aout = self.ahead(fea)
        apred = torch.tanh(aout)
        
        return au_out, expr_out, vpred, apred
