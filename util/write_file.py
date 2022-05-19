"""
Name: write_file
Date: 19-10-8 上午8:37
"""

import os
import random
import time


def WriteFile(file_dir, file_name, file_content, file_mode, change_file_name=False):
    """
    :param file_dir:
    :param file_name:
    :param file_content:
    :param file_mode:
    :param change_file_name:  这个主要是针对创建checkpoint文件夹的时候，如果出现了相同的文件夹的名字，那么就自动修改创建的文件夹的时间，防止出现两个程序创建了相同的文件夹
    :return:
    """
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    elif os.path.exists(file_dir) and change_file_name:
        while os.path.exists(file_dir):
            # 这部分的代码改变方式只适合于55-2021-04-28-14-35-09-single-3-，同时这里改变的是毫秒
            file_dir_list = file_dir.split('-')
            file_dir_list[6] = str((int(file_dir_list[6]) + random.randint(61, 119)) % 60)
            if len(file_dir_list[6]) == 1:
                file_dir_list[6] = '0' + file_dir_list[6]
            file_dir = '-'.join(file_dir_list)
        os.mkdir(file_dir)
    f = open(file_dir + '/' + file_name, file_mode, encoding='utf-8')
    f.write(file_content)
    f.close()
    return file_dir


# file_dir = '55-2021-04-28-14-35-09-single-3-'
# WriteFile(file_dir, 'train_correct_log.txt', 'str(opt)+aaaaaaaaaaaaaaaaaaaaaaaa' + str(time.time()) + '\n\n', 'a+', change_file_name=True)