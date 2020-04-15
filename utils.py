# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:13:12 2020

@author: Ulrich
"""


def load_classes(path):
    """
    Load class labels from the assigned path.
    
    Input:
        path - repository of class file.
    Return:
        classes - class labels
    """
    
    file = open(path, 'r')
    classes = file.read().split('\n')[:-1]
    
    return classes
    
def non_max_suppression(detections, num_classes, conf_threshold=0.5, nms_threshold=0.4):
    """
    Remove the detection results with lower confidence score than conf_threshold.
    Apply non-maximum suppression to filter detetions results.
    
    Input:
        detections - all detected result from the model with the specific image.
        num_classes - the number of classes in the current dataset.
        conf_threshold
        nms_threshold
    """
    
    output = [None for _ in range(len(detections))]
    
    return output

    
    
    