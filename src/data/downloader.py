# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 19:40:19 2026

@author: Ding Zhang
"""

import torchvision
import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))

dataset = torchvision.datasets.OxfordIIITPet(root='./raw',split='trainval',target_types='category',download=True)

class_name = dataset.classes
image,idx = dataset[1]
