# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from collections import defaultdict

from tools.parse_config import parse_model_config
from tools.utils import build_targets

def generate_modules(layers):
    """
    Generate a module list from the layers with configurations.
    
    Input:
        layers - A list of layers with a dictionary for configurations.
    Return:
        hyperparams
        module_list
    """
    hyperparams = layers.pop(0)
    output_filters = [int(hyperparams['channels'])]
    
    module_list = nn.ModuleList()
    for i, layer in enumerate(layers):
        modules = nn.Sequential()
        
        if layer['type'] == 'convolutional':
            bn = int(layer['batch_normalize'])
            filters = int(layer['filters'])
            kernel_size = int(layer['size'])
            pad = (kernel_size-1) // 2 if int(layer['pad']) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(layer['stride']),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if layer['activation'] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))
            
        elif layer['type'] == 'maxpool':
            kernel_size = int(layer['size'])
            stride = int(layer['stride'])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)
            
        elif layer['type'] == 'upsample':
            stride = int(layer['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='nearest')
            modules.add_module("upsample_%d" % i, upsample)
            
        elif layer['type'] == 'route':
            layer = [int(x) for x in layer['layers'].split(',')]
            filters = sum([output_filters[layer_idx] for layer_idx in layer])
            modules.add_module("route_%d" % i, EmptyLayer())
            
        elif layer['type'] == 'shortcut':
            filters = output_filters[int(layer['from'])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())
            
        elif layer['type'] == 'yolo':
            num_classes = int(layer['classes'])
            img_height = int(hyperparams['height'])
            
            # Extract anchors.
            anchor_idx = [int(x) for x in layer['mask'].split(',')]
            anchors = [int(x) for x in layer['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idx]
            
            # YOLO detection layer.
            yolo_layer = YOLOLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)
        
        # Add the module seqnence into module list.
        module_list.append(modules)
        output_filters.append(filters)
    
    return hyperparams, module_list

class Darknet(nn.Module):
    """
    YOLOv3 object detection model.
    """
    
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.layers = parse_model_config(config_path)
        self.hyperparams, self.module_list = generate_modules(self.layers)
        
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]
        
    def forward(self, x, targets=None):
        is_training = targets is not None
        
        self.losses = defaultdict(float)
        output = []
        layer_outputs = []
        
        # Customize route, shortcut and yolo layer.
        for i, (layer, module) in enumerate(zip(self.layers, self.module_list)):
            if layer['type'] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
                
            elif layer['type'] == 'route':
                layer_idx = [int(x) for x in layer['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_idx], 1)
                
            elif layer['type'] == 'shortcut':  
                layer_idx = int(layer["from"])
                x = layer_outputs[-1] + layer_outputs[layer_idx]
                
            elif layer['type'] == 'yolo':
                # Training.
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Testing.
                else:
                    x = module(x)
                output.append(x)
                
            # Add to the list whatever it is.    
            layer_outputs.append(x)
        
        self.losses['recall'] /= 3
        self.losses['precision'] /= 3
        return sum(output) if is_training else torch.cat(output, 1)
        
    
    def load_weights(self, weights_path):
        """
        Load pre-trained weights.
        """
        
        # Load weights from the file.
        fp = open(weights_path, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)   # First five are header values.
        self.header_info = header   # Save it for writing.
        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()
        
        # Parsing the weights from layers.
        ptr = 0
        for i, (layers, module) in enumerate(zip(self.layers, self.module_list)):
            if layers['type'] == 'convolutional':
                conv_layer = module[0]
                if layers['batch_normalize']:       
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                            
                    # Bias
                    num_b = bn_layer.bias.numel()  # Number of biases
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load convolutional layer bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                    
                # Load convolutional layer weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
        
    def save_weights(self, path, cutoff=-1):
        fp = open(path, 'wb')
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)
        
        for i, (layers, module) in enumerate(zip(self.layers[: cutoff], self.module_list[: cutoff])):
            if layers['type'] == 'convolutional':
                conv_layer = module[0]
                # Check whether it has batch normalization.
                if layers['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        
        fp.close()
        
        
class EmptyLayer(nn.Module):
    """
    Placeholder for 'route' and 'shortcut' layers.
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()
        
        
class YOLOLayer(nn.Module):
    """
    YOLO detection layer.
    """      
    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.image_dim = img_dim
        
        self.bbox_attrs = 5 + num_classes
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)   # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)   # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()            # Class loss
    
    def forward(self, x, targets=None):
        num_anchor = self.num_anchors
        nB = x.size(0)
        grid_size = x.size(2)
        stride = self.image_dim / grid_size
        
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor
        
        prediction = x.view(nB, num_anchor, self.bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
        
        # Extract outputs.
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        weight = prediction[..., 2]
        height = prediction[..., 3]
        pred_conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])
        
        # Calculate offsets for each grid.
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(FloatTensor)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, num_anchor, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, num_anchor, 1, 1))
        
        # Offset adjustment and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(weight.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(height.data) * anchor_h
        
        # Training
        if targets is not None:
            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()
            
            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=num_anchor,
                num_classes=self.num_classes,
                grid_size=grid_size,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            # Confused matrix.
            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals) if nGT else 1
            
            # Handle masks and target variables.
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)
            
            # Mask outputs to ignore non-existing objects
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(weight[mask], tw[mask])
            loss_h = self.mse_loss(height[mask], th[mask])
            
            loss_conf_false = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false])
            loss_conf_true = self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss_conf = loss_conf_false + loss_conf_true
                    
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            
            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )
            
        # Testing.
        else:
            return torch.cat((pred_boxes.view(nB, -1, 4) * stride,
                              pred_conf.view(nB, -1, 1),
                              pred_cls.view(nB, -1, self.num_classes)), -1)
        
        
        
        