#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os

# For every 10 images, the first nine are used as training data while the last one is used as validation data.
cur_dir = "./data/images"
split_pct = 10
idx_test = round(100 / split_pct)
file_train = open("data/train.txt", 'w')
file_val = open("data/val.txt", 'w')
cnt = 1

for file_path in glob.iglob(os.path.join(cur_dir, "*.jpg")):
    name, suffix = os.path.splitext(os.path.basename(file_path))
    if cnt == idx_test:
        cnt = 1
        file_val.write(cur_dir + "/" + name + ".jpg" + "\n")
    else:
        file_train.write(cur_dir + "/" + name + ".jpg" + "\n")
        cnt += 1

file_train.close()
file_val.close()
