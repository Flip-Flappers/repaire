import os
import shutil
from glob import glob



for tar_c in range(10):
    num = 0
    for c in range(10):
        if c == tar_c:
            for i in range(50):
                s = "{:04d}".format(i)

                shutil.copy('../division/pngcifar10/test/' + str(tar_c) + '/' + str(i) + '.png' , './test_data_type_D/divide_' + str(tar_c) + '/pic_A/' + s + '.png')

        else:
            for i in range(50):
                if i % 10 == 0:
                    s = "{:04d}".format(num)

                    shutil.copy('../division/pngcifar10/test/' + str(c) + '/' + str(i) + '.png',
                               './test_data_type_D/divide_' + str(tar_c) + '/pic_B/' + s + '.png')
                    num += 1