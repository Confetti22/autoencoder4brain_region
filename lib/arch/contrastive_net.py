import torch.nn as nn
import torch.nn.functional as F

from lib.arch.block import *
from lib.arch.utils import model_init
from lib.arch.utils import  get_activation
from .encoder import Encoder
class Contrastive_net(nn.Module):
    def __init__(self,
                 in_channel: int = 1,
                 input_size: int = [128,128,128],
                 encdoer_filters: List[int] = [32,64,96],
                 encoder_block_type: str = 'single',
                 pooling_with_conv: List[bool] = [True,True],
                 fc_layer: bool  = True,
                 emb_dim: int = 512,
                 decoder_filters:List[int] = [64,64,64,64,64],
                 projection_dim : int = 64 ,
                 pad_mode: str = 'reflect',
                 act_mode: str = 'elu',
                 norm_mode: str = 'gn',
                 init_mode: str = 'orthogonal',
                 **kwargs
              
               ):
        super().__init__()
        
        self.fc_layer = fc_layer
        self.emb_dim= emb_dim 
        self.encoder = Encoder(in_channel,encdoer_filters,pad_mode,act_mode,norm_mode,
                               init_mode=init_mode,block_type=encoder_block_type,**kwargs)
        
        self.sum_layer = nn.ModuleList()
        for i in range(len(pooling_with_conv)):
            if pooling_with_conv[i]:
                self.sum_layer.append(
                    nn.Sequential(
                        nn.AvgPool3d(kernel_size=2,stride=2),
                        nn.Conv3d(encdoer_filters[-1],encdoer_filters[-1],kernel_size=3,stride=1,padding=1,padding_mode=pad_mode)
                    )
                )
            else :
                self.sum_layer.append(
                    nn.Sequential(
                        nn.AvgPool2d(kernel_size=2,stride=2)
                    )
                )
        
        if fc_layer:
            enocder_min_spatio_len = int( input_size[0] /( 2**(len(encdoer_filters))) )
            min_spatio_len = int( enocder_min_spatio_len / (2**(len(pooling_with_conv))))
            self.flattened_dim = encdoer_filters[-1]*(min_spatio_len)**3
            self.fc1 = nn.Linear(self.flattened_dim, self.emb_dim)        
            self.fc2= nn.Linear(self.emb_dim, self.flattened_dim)


        self.contras_decoder = nn.ModuleList()
        for i in range(len(decoder_filters) -1 ):
            if i == 0:
                previous_filters=encdoer_filters[-1]
            else :
                previous_filters= decoder_filters[i-1]
            self.contras_decoder.append(
                transconv3d_norm_act(previous_filters,decoder_filters[i],kernel_size=3,
                                     stride=2,output_padding=1,padding=1,norm_mode='none',act_mode='elu')
            )
        self.contras_decoder.append(
                          transconv3d_norm_act(previous_filters,decoder_filters[i],kernel_size=5,
                                     stride=2,output_padding=1,padding=2,norm_mode='none',act_mode='elu')

        )
        self.contrastive_projt = nn.Sequential(
                nn.Conv3d(in_channels=decoder_filters[-1],out_channels=projection_dim,kernel_size=1,stride=1),
                get_activation(act_mode),
                nn.Conv3d(in_channels=projection_dim,out_channels=projection_dim,kernel_size=1,stride=1),
                )

    def forward(self,x) :
        #encoder_trained_from_autoencoder
        x = self.encoder(x)
        #summary
        for i in range(len(self.sum_layer)):
            x = self.sum_layer[i](x)

        #whether have the fc layer
        if self.fc_layer:
            batch_size = x.shape[0]
            self.shape_before_flattening = x.shape[1:]

            flattened_x = x.view(batch_size,-1) 
            embd_vector = self.fc1(flattened_x) 
            flattened_x2=self.fc2(embd_vector)
            x = flattened_x2.view(batch_size,*self.shape_before_flattening)
                
        
        #contrastive_decoder path
        for i in range(len(self.contras_decoder)):
            x = self.contras_decoder[i](x)
        
        #last contrastive layer
        x = self.contrastive_projt(x) 
        return x

MODEL_MAP = {
    'contrastive_net': Contrastive_net,
}

def build_contrastive_net(cfg):

    model_arch = cfg.MODEL.ARCHITECTURE
    assert model_arch in MODEL_MAP.keys()
    kwargs = {
        'in_channel': cfg.MODEL.IN_PLANES,
        'input_size': cfg.MODEL.INPUT_SIZE,
        'encoder_filters': cfg.MODEL.FILTERS,
        'encoder_block_type': cfg.MODEL.BLOCK_TYPE,
        'pooling_with_conv': cfg.MODEL.pooling_with_conv,
        'fc_layer':cfg.MODEL.fc_layer,
        'pad_mode': cfg.MODEL.PAD_MODE,
        'act_mode': cfg.MODEL.ACT_MODE,
        'norm_mode': cfg.MODEL.NORM_MODE,
    }

    model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](**kwargs)
    print('model: ', model.__class__.__name__)

    return model
        

            



