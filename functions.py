#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:49:40 2019
@author: will
"""
import matplotlib.pyplot as plt
import albumentations as albu
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_value_
from pytorch_util import set_requires_grad
import numpy as np
import time
import copy
#from segmentation_models_pytorch.base.encoder_decoder import EncoderDecoder
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import relu_fn,load_pretrained_weights,get_model_params,drop_connect,get_same_padding_conv2d
from segmentation_models_pytorch.utils.losses import BCEDiceLoss
from apex import amp
import cv2

'''------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------------ Data -----------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''

preprocessing_dict = {256:(0.5506135154400696, 0.25807634194916107)}

encoder_channels={'efficientnet-b0': [320, 112, 40, 24, 16],
                  'efficientnet-b1': [320, 112, 40, 24, 16],
                  'efficientnet-b2': [352, 120, 48, 24, 16],
                  'efficientnet-b3': [384, 136, 48, 32, 24],
                  'efficientnet-b4': [448, 160, 56, 32, 24],
                  'efficientnet-b5': [512, 176, 64, 40, 24],
                  'efficientnet-b6': [576, 200, 72, 40, 32],
                  'efficientnet-b7': [640, 224, 80, 48, 32]}

# class dataset_old(Dataset):
#     # this load all images once. does not work for 1024*1024
#     def __init__(self, images,masks,augmentation=None,preprocessing=None):
#         # preprocessing should be a tuple (mean,std) used to normalize image
#         self.images = images
#         self.masks = masks
#         self.augmentation = augmentation
#         self.preprocessing = preprocessing
    
#     def _cal_preprocessing(self):
#         n = len(self)
#         mean,std = 0,0
#         for i in range(n):
#             image = self.images[i]
#             image = self.augmentation(image=image)['image']
#             mean = mean + image.mean()
#             std = std + (image**2).mean()
#         mean = mean/n
#         std = np.sqrt(std/n-mean**2)
#         self.preprocessing = mean,std
#         print(mean,std)
    
#     def __getitem__(self, i):
#         image = self.images[i]
#         mask = self.masks[i]
        
#         # apply augmentations
#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask)
#             image, mask = sample['image'], sample['mask']
        
#         # apply preprocessing
#         if self.preprocessing:
#             image = (image-self.preprocessing[0])/self.preprocessing[1]
            
#         return image[None], mask[None]
        
#     def __len__(self):
#         return self.images.shape[0]

class dataset(Dataset):
    def __init__(self, imageId,images_dir,masks_dir=None,augmentation=None,preprocessing=None):
        # preprocessing should be a tuple (mean,std) used to normalize image
        self.imageId = imageId
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def _cal_preprocessing(self):
        n = len(self)
        mean,std = 0,0
        for i in range(n):
            image = np.load(self.images_dir+self.imageId[i]+'.npy')
            image = self.augmentation(image=image)['image']
            mean = mean + image.mean()
            std = std + (image**2).mean()
        mean = mean/n
        std = np.sqrt(std/n-mean**2)
        self.preprocessing = mean,std
        print(mean,std)
    
    def __getitem__(self, i):
        image = np.load(self.images_dir+self.imageId[i]+'.npy')
        if self.masks_dir is not None:
            mask = np.load(self.masks_dir+self.imageId[i]+'.npy')
            # apply augmentations
            if self.augmentation:            
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            # apply preprocessing
            if self.preprocessing:
                image = (image-self.preprocessing[0])/self.preprocessing[1]
            return image[None], mask[None]
        else:
            # apply augmentations
            if self.augmentation:            
                sample = self.augmentation(image=image)
                image = sample['image']
            # apply preprocessing
            if self.preprocessing:
                image = (image-self.preprocessing[0])/self.preprocessing[1]
            return image[None],     
        
    def __len__(self):
        return len(self.imageId)


transform = albu.Compose([albu.HorizontalFlip(),
                          albu.OneOf([albu.RandomContrast(),
                                      albu.RandomGamma(),
                                      albu.RandomBrightness(),
                                      ], p=0.3),
                          albu.OneOf([albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                                      albu.GridDistortion(),
                                      albu.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                                      ], p=0.15),
                          albu.ShiftScaleRotate()])
    
'''------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------- Model -----------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''
    
class EfficientNet_encoder(EfficientNet):
    """
    differs from EfficientNet in that forward outputs a list of tensors for Unet and does not have fc
    
    """
    def __init__(self, blocks_args=None, global_params=None,includeX0=False):
        super().__init__(blocks_args,global_params)
        # delete linear layer
        del self._dropout
        del self._fc
        del self._bn1
        del self._conv_head
        self.includeX0 = includeX0
        self._special_layers()
        
    def _special_layers(self):
        self._layers = np.where([i._depthwise_conv.stride[0]==2 for i in self._blocks])[0]

    def forward(self, inputs):
        """ return a list of tensor each half in size as previous one
            e.g. (5,16,128,128) -> (5,24,64,64)...
        """
        x_list = [inputs if self.includeX0 else None]
        x = relu_fn(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            if idx in self._layers:
                x_list.append(x)
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        
        x_list.append(x)
        return x_list 

    @classmethod
    def from_name(cls, model_name, override_params=None,**kways):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet_encoder(blocks_args, global_params,**kways)

    @classmethod
    def from_pretrained(cls, model_name, override_params=None,**kways):
        model = EfficientNet_encoder.from_name(model_name,override_params,**kways)
        load_pretrained_weights(model, model_name, load_fc=False)
        return model

class MBConvBlock(nn.Module):
    """
    remove block_args (namedtuple): and global_params (namedtuple)
    """
    def __init__(self,input_filters,output_filters,image_size,expand_ratio,
                 kernel_size=3,stride=1,has_se=True,id_skip=True,se_ratio=0.25):
        super().__init__()
        self._bn_mom = 1 - 0.99
        self._bn_eps = 0.001
        self.has_se = has_se
        self.id_skip = id_skip
        self.stride = stride
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.expand_ratio = expand_ratio
        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Expansion phase
        inp = input_filters  # number of input channels
        oup = input_filters * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = kernel_size
        s = stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(input_filters * se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))
        # Skip connection and drop connect
        if self.id_skip and self.stride == 1 and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

class UpConcatBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return x
    
class EfficientNet_decoder(nn.Module):

    def __init__(
            self,
            image_size,
            encoder_channels,
            expand_ratio=6,
            decoder_channels=(256, 128, 64, 32, 16),
            decoder_repeats=(2,2,2,2,2),
            final_channels=1
    ):
        super().__init__()

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = nn.Sequential(*[UpConcatBlock(),MBConvBlock(in_channels[0],out_channels[0],image_size,expand_ratio)]+
                                    [MBConvBlock(out_channels[0],out_channels[0],image_size,expand_ratio) for _ in range(decoder_repeats[0])])
        self.layer2 = nn.Sequential(*[UpConcatBlock(),MBConvBlock(in_channels[1],out_channels[1],image_size,expand_ratio)]+
                                    [MBConvBlock(out_channels[1],out_channels[1],image_size,expand_ratio) for _ in range(decoder_repeats[1])])
        self.layer3 = nn.Sequential(*[UpConcatBlock(),MBConvBlock(in_channels[2],out_channels[2],image_size,expand_ratio)]+
                                    [MBConvBlock(out_channels[2],out_channels[2],image_size,expand_ratio) for _ in range(decoder_repeats[2])])
        self.layer4 = nn.Sequential(*[UpConcatBlock(),MBConvBlock(in_channels[3],out_channels[3],image_size,expand_ratio)]+
                                    [MBConvBlock(out_channels[3],out_channels[3],image_size,expand_ratio) for _ in range(decoder_repeats[3])])
        self.layer5 = nn.Sequential(*[UpConcatBlock(),MBConvBlock(in_channels[4],out_channels[4],image_size,expand_ratio)]+
                                    [MBConvBlock(out_channels[4],out_channels[4],image_size,expand_ratio) for _ in range(decoder_repeats[4])])
        
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))
#        self.initialize()
#        
#    def initialize(self):
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#            elif isinstance(m, nn.BatchNorm2d):
#                nn.init.constant_(m.weight, 1)
#                nn.init.constant_(m.bias, 0)
                
    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        x = x[::-1]
        encoder_head = x[0]
        skips = x[1:]

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, skips[4]])
        x = self.final_conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, encoder, decoder,activation='sigmoid'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = BCEDiceLoss(activation=activation)
        
        if callable(activation):
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid" or "softmax"')
            
    def forward(self,x,mask=None):
        x = x.expand(-1,3,-1,-1) # gray to RGB
        x = self.encoder(x)
        x = self.decoder(x)
        if mask is not None:
            return self.loss(x,mask)
        else:
            return x

    def predict(self,x,threshold=None):
        """Inference method. """
        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)
            if threshold is not None:
                x = (x > threshold).float()
        return x
    
'''------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------------- utility -----------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------'''

def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start+index
        end = start+length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component

def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0]+1
    end = np.where(component[:-1] > component[1:])[0]+1
    length = end-start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i]-end[i-1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle

def predict(model,loader):
    model.eval()
    yhat_list = []
    with torch.no_grad():
        for data in loader:
            data = [out.to('cuda:0') for out in data]
            yhat_list.append(model.predict(*data).cpu().detach().numpy())
    yhat = np.concatenate(yhat_list)
    return yhat.squeeze()

def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros_like(probability)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num

def submit(df,yhat,best_threshold, min_size,name):
    encoded_pixels = []
    for probability in yhat:
        if probability.shape != (1024, 1024):
            probability = cv2.resize(probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR)
        predict, num_predict = post_process(probability, best_threshold, min_size)    
        if num_predict == 0:
            encoded_pixels.append('-1')
        else:
            r = run_length_encode(predict)
            encoded_pixels.append(r)
    df['EncodedPixels'] = encoded_pixels
    df.to_csv(name, columns=['ImageId', 'EncodedPixels'], index=False)
    
def visualize(image,mask):
    plt.imshow(image)
    plt.imshow(mask,alpha=0.5)

def train(opt,model,epochs,train_dl,val_dl,paras,clip,\
          scheduler=None,patience=6,saveModelEpoch=9):
    # add early stop for 5 fold
    since = time.time()
    counter = 0 
    lossBest = 1e6
    bestWeight,bestOpt,bestAmp = [None]*3
        
    opt.zero_grad()
    for epoch in range(epochs):
        # training #
        model.train()
        train_loss = 0
        val_loss = 0
        
        for i,data in enumerate(train_dl):
            data = [out.to('cuda:0') for out in data]
            loss = model(*data)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            clip_grad_value_(amp.master_params(opt),clip)
            opt.step()
            opt.zero_grad()
            train_loss += loss.item()
        train_loss = train_loss/i
        
        # evaluating #
        model.eval()
        with torch.no_grad():
            for j,data in enumerate(val_dl):
                data = [out.to('cuda:0') for out in data]
                loss = model(*data)
                val_loss += loss.item()
        val_loss = val_loss/j
        
        # save model
        if val_loss<lossBest:
            lossBest = val_loss
            if epoch>saveModelEpoch:
                bestWeight = copy.deepcopy(model.state_dict())
                bestOpt = copy.deepcopy(opt.state_dict())
                bestAmp = copy.deepcopy(amp.state_dict())
                
        print('epoch:{}, train_loss: {:+.3f}, val_loss: {:+.3f}\n'.format(epoch,train_loss,val_loss))
        if scheduler is not None:
            scheduler.step(val_loss)
                
        # early stop
        if val_loss==lossBest:
            counter = 0
        else:
            counter+= 1
            if counter >= patience:
                print('----early stop at epoch {}----'.format(epoch))
                time_elapsed = time.time() - since
                print('Training completed in {}s'.format(time_elapsed))
                return model,bestWeight,bestOpt,bestAmp
            
    time_elapsed = time.time() - since
    print('Training completed in {}s'.format(time_elapsed))
    return model,bestWeight,bestOpt,bestAmp

# for j in range(8):
#     print('model:{}'.format(j))
#     model = EfficientNet.from_name('efficientnet-b'+str(j))
#     for arg in model._blocks_args:
# print(arg.expand_ratio)