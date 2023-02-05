import numpy
import math

import numpy as np
from PIL import Image
from tqdm import tqdm

loc = []
for i in range(6):
    for j in range(6):
        loc.append([i, j])

num = int(math.pow(2, 36))
print(num)
for i in tqdm(range(num)):
    tmp_str = bin(i)[2:]
    tmp_image = np.zeros([6, 6])
    for j in range(len(tmp_str)):
        if tmp_str[j] == '1':
            tmp_image[loc[j][0]][loc[j][1]] = 1
    im = Image.fromarray(np.uint8(255 * tmp_image))
    im.save('../../recover_dataset/own_dic/' + str(i) + '.png');
