import torch.nn as nn
import torch.nn.functional as F

from lib.arch.block import *
from .encoder import Encoder

class Classifier(nn.Module):
    def __init__(self,
                 in_channel: int = 1,
                 input_size: List[int] =[128,128,128],
                 encoder_filters: List[int] = [32,64,96],
                 encoder_block_type: str = 'single',
                 pad_mode: str = 'reflect',
                 act_mode: str = 'elu',
                 norm_mode: str = 'none',
                 encoder_kernel_size:List[int] =[5,3,3],
                 sum_layer_type :str = 'adaptive',
                 init_mode: str = 'none',
                 num_class: int = 2,
                 **kwargs
                 ):
        super().__init__()
        self.cnn= Encoder(in_channel,encoder_filters,
                          pad_mode,act_mode,norm_mode,kernel_size=encoder_kernel_size,init_mode=init_mode,
                          block_type=encoder_block_type,
                          **kwargs)
        

        # each layer of encoder reduce spatial dimention by 2, and sum_layer will reduce by 4
        # min_spatio_len = input_size[0]//(2**( len(encoder_filters)+2 ))
        min_spatio_len = 1  #using average/gaussian in sum_layer to reduce the spatio_dim into 1
        print(f"min_spatio_len in featuremap after sum_layer is {min_spatio_len}")
        filters_num = encoder_filters[-1]
        if sum_layer_type == 'conv':
            self.sum_layer = nn.Sequential(
                nn.Conv3d(filters_num,filters_num,
                          kernel_size=3,stride=2,padding=1),
                nn.ELU(),
                nn.Conv3d(filters_num,filters_num,
                          kernel_size=3,stride=2,padding=1),
                nn.ELU(),
                nn.AdaptiveAvgPool3d(output_size=1)
            )

        elif sum_layer_type =='conv_pool':
            self.sum_layer = nn.Sequential(
                nn.Conv3d(filters_num,filters_num,
                          kernel_size=3,stride=1,padding=1),
                nn.AvgPool3d(kernel_size=2,stride=2),
                nn.ELU(),
                nn.Conv3d(filters_num,filters_num,
                          kernel_size=3,stride=1,padding=1),
                nn.AvgPool3d(kernel_size=2,stride=2),
                nn.ELU(),
            )

        else:
            self.sum_layer = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=min_spatio_len)
             )


        flattened_dim = (encoder_filters[-1])*(min_spatio_len)**3

        self.fc = nn.Sequential(
            nn.Linear(flattened_dim,int(flattened_dim//2)),
            nn.ReLU(),
            nn.Linear(int(flattened_dim//2), num_class),
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(flattened_dim,num_class),
        # )
    
    def forward(self,x):
        x = self.cnn(x)
        x = self.sum_layer(x)
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        return x

MODEL_MAP ={
    'classifier': Classifier,
}

def build_classifier(cfg):

    model_arch = cfg.MODEL.ARCHITECTURE
    assert model_arch in MODEL_MAP.keys()
    kwargs = {
        'in_channel': cfg.MODEL.IN_PLANES,
        'input_size': cfg.DATASET.input_size,
        'encoder_filters': cfg.MODEL.FILTERS,
        'encoder_kernel_size':cfg.MODEL.kernel_size,
        'encoder_block_type': cfg.MODEL.BLOCK_TYPE,
        'pooling_with_conv': cfg.MODEL.pooling_with_conv,
        'sum_layer_type':cfg.MODEL.sum_layer,
        'pad_mode': cfg.MODEL.PAD_MODE,
        'act_mode': cfg.MODEL.ACT_MODE,
        'norm_mode': cfg.MODEL.NORM_MODE,
    }

    model = MODEL_MAP[cfg.MODEL.ARCHITECTURE](**kwargs)
    print('model: ', model.__class__.__name__)

    return model
        

            



