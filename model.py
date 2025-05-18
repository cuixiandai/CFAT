import numpy as np
import torch
from torch.nn.functional import relu
from torch.nn import LayerNorm,Linear,Dropout
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder

oc=4
encoder_in=64
dim_f=64
L1O=128
png_in=49

def position_embeddings(n_pos_vec, dim):
    position_embedding = torch.nn.Embedding(n_pos_vec.numel(), dim)
    torch.nn.init.constant_(position_embedding.weight, 0.)
    return position_embedding

class MyModel(torch.nn.Module):
    def __init__(self,d_model=oc*2*7*7,num_encoder_layers=5,nhead=8,dropout=0.1,dim_feedforward=dim_f,batch_size=32):
        super(MyModel, self).__init__()
        #self.conv1 = torch.nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1) 
        self.convR = torch.nn.Conv2d(in_channels=6, out_channels=oc, kernel_size=3, stride=2, padding=1)
        self.convI = torch.nn.Conv2d(in_channels=6, out_channels=oc, kernel_size=3, stride=2, padding=1)

        self.embedding_layers = torch.nn.ModuleList([Linear(png_in, encoder_in) for _ in range(oc*2*2)])

        encoder_layer = TransformerEncoderLayer(encoder_in, nhead, dim_feedforward, dropout,norm_first=False)#encoder_in
        encoder_norm =LayerNorm(encoder_in)#d_model
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.linear0=Linear(d_model*2,encoder_in)
        #self.linear0=Linear(49,768)
        self.linear1=Linear(encoder_in*8*2, L1O)
        self.linear2=Linear(L1O, 15)   
        self.position_embedding = position_embeddings(torch.arange(batch_size*d_model*2), 1)
        self.d_model=d_model

        self.dropout1 = Dropout(p=0.25) 
        self.dropout2 = Dropout(p=0.1) 

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, encoder_in))
        torch.nn.init.normal_(self.cls_token)   

    def forward(self, src):
        batch_size=src.shape[0]
        valueR=relu(self.convR(src[:,0:6,:,:]))
        valueI=relu(self.convI(src[:,6:12,:,:]))
        value=torch.cat((valueR,valueI),1)
        value2=torch.transpose(value, 2, 3)
        value=torch.cat((value,value2),1)
        position_ids = torch.arange(src.shape[0] * self.d_model*2, dtype=torch.long, device=src.device)  
        position_embeds = self.position_embedding(position_ids).view(batch_size, self.d_model*2)    
        value=torch.reshape(value,(src.shape[0],-1))
        value = value + position_embeds

        value=value.view(batch_size,oc*2*2,-1)

        embedded_values = []
        for i in range(oc*2*2):
            embedded_value = self.embedding_layers[i](value[:, i, :])
            embedded_values.append(embedded_value)
        value = torch.stack(embedded_values, dim=1)

        value=self.encoder(value)
        value=value.view(batch_size,-1)

        value=value.squeeze(1)
        value=relu(self.linear1(value))

        value=self.linear2(value)
        return value