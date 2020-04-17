# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:52:54 2020

@author: Ulrich
"""


def parse_model_config(path):
    """
    Parses the YOLOv3 layer configuration and returns the module layers.
    
    Example:
        [convolutional]
        batch_normalize=1
        filters = 32
        size=3
    The type name of layer is surrounded by a block. The whitespaces around '=' or on the right of values need to be removed.

    Input: 
        path - configuration file repository.
    Return:
        layers - layers for YOLOv3.
    """
    cfg_file = open(path, 'r')
    lines = cfg_file.read().split('\n')
    
    # Skip the commented lines.
    lines = [x for x in lines if x and not x.startswith('#')]
    # Ignore the fringe whitespaces.                                                  
    lines = [x.rstrip().lstrip() for x in lines]

    layers = []
    for line in lines:
        # A new block for new layer.
        if line.startswith('['):
            layers.append({})
            layers[-1]['type'] = line[1:-1].rstrip()   # Skip the block '[' and ']'.
            
            # Some convolutional layer might have no batch normalization process.
            if layers[-1]['type'] == "convolutional":
                layers[-1]['batch_normalize'] = 0  
        else:
            key, value = line.split('=')
#            value = value.strip()
            layers[-1][key.rstrip()] = value.strip()
        
    return layers


