import torch.nn as nn
from lib.arch.block import *

class Encoder(nn.Module):

    block_dict = {
        'single': SingleConv3d,
        'double': DoubleConv3d, 
        'residual':BasicBlock3d,
    }

    def __init__(self,
                 in_channel: int = 1,
                 filters: List[int] = [32,64,96],
                 pad_mode: str = 'reflect',
                 act_mode: str = 'elu',
                 norm_mode: str = 'none',
                 kernel_size:List[int] =[5,3,3],
                 init_mode: str = 'orthogonal',
                 block_type: str = 'single',
                 **kwargs):
        super().__init__()

        self.filters = filters
        self.kernel_size =kernel_size
        self.depth = len(filters)


        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode}

        stride = 2
        padding = int(( int( self.kernel_size[0] ) -1)//2)
        print(f"padding ={padding},k{self.kernel_size[0]}")
        # self.conv_in =nn.Sequential( 
        #     nn.Conv3d(in_channel,int(filters[0]//2),kernel_size=self.kernel_size[0],stride =stride,padding = padding),
        #     conv3d_norm_act(int(filters[0]//2), filters[0], kernel_size=self.kernel_size[0],
        #                                stride=1,padding=padding, **self.shared_kwargs)
        #     )
        self.conv_in =conv3d_norm_act(in_channel, filters[0], kernel_size=self.kernel_size[0],
                                       stride=stride,padding=padding, **self.shared_kwargs)
        # encoding path
        self.down_layers = nn.ModuleList()
        for i in range(self.depth -1):
            next = min(self.depth, i+1)
            kernel_size = self.kernel_size[next]
            stride = 2
            padding = int((kernel_size -1)//2) 

            if block_type == 'single':
                self.down_layers.append(
                  nn.Sequential(
                      conv3d_norm_act(filters[i],filters[next],kernel_size=kernel_size,stride = stride, padding=padding,**self.shared_kwargs),
                      conv3d_norm_act(filters[next],filters[next],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs)
                     )
                )

            elif block_type == 'double':
                self.down_layers.append(
                    nn.Sequential(
                        conv3d_norm_act(filters[i],filters[next],kernel_size=kernel_size,stride = stride, padding=padding,**self.shared_kwargs),
                        conv3d_norm_act(filters[next],filters[next],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs),
                        conv3d_norm_act(filters[next],filters[next],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs)
                         )
                )
            elif block_type == 'residual' :
                self.down_layers.append(
                    nn.Sequential(
                        conv3d_norm_act(filters[i],filters[next],kernel_size=kernel_size,stride = stride, padding=padding,**self.shared_kwargs),
                        ResidualBlock3d(filters[next],filters[next],kernel_size=kernel_size,stride = 1, padding=padding,**self.shared_kwargs)
                          )
                )
            else:
                self.down_layers.append(
                    nn.Sequential(
                        conv3d_norm_act(filters[i],filters[next],kernel_size=kernel_size,stride = stride, padding=padding,**self.shared_kwargs),
                          )
                )
 

        #linear projection for embdding
        self.last_encoder_conv=nn.Conv3d(self.filters[-1],self.filters[-1],kernel_size=1,stride=1)

       
        # initialization
        # model_init(self, mode=init_mode)

    def forward(self, x):
        
        #encoder path
        x = self.conv_in(x)
        for i in range(self.depth-1):
            x = self.down_layers[i](x)
        x = self.last_encoder_conv(x) 
        return x

