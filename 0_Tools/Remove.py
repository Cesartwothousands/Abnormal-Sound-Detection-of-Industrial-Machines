from Pre_Data import *
import os
import shutil
import time

time_start = time.time()  # time = 0
path_file = r'F:\毕业论文\Dataset'
path_file_new = r'F:\毕业论文\Pictures\MIMII_wav'
type = '.wav'

for i in range(2,4):
    number_ab = 0
    number_no = 0
    path_file_1 = ''
    path_file_new_1 = ''

    if i == 0 :
        path_file_1 = path_file + r'\-6_dB_valve\valve'
        path_file_new_1 = path_file_new + r'\valve'
    elif i == 1 :
        path_file_1 = path_file + r'\-6_dB_slider\slider'
        path_file_new_1 = path_file_new + r'\slider'
    elif i == 2 :
        path_file_1 = path_file + r'\-6_dB_pump\pump'
        path_file_new_1 = path_file_new + r'\pump'
    elif i == 3 :
        path_file_1 = path_file + r'\-6_dB_fan\fan'
        path_file_new_1 = path_file_new + r'\fan'

    for ii in range(4):
        path_file_2 = ''

        if ii == 0:
            path_file_2 = path_file_1 + r'\id_00'
        elif ii == 1:
            path_file_2 = path_file_1 + r'\id_02'
        elif ii == 2:
            path_file_2 = path_file_1 + r'\id_04'
        elif ii == 3:
            path_file_2 = path_file_1 + r'\id_06'

        for iii in range(2):
            path_file_3 = ''
            path_file_new_3 = ''

            if iii == 0:
                path_file_3 = path_file_2 + r'\abnormal'
                path_file_new_3 = path_file_new_1 + r'\abnormal'
            else:
                path_file_3 = path_file_2 + r'\normal'
                path_file_new_3 = path_file_new_1 + r'\normal'

            if iii == 0:
                all_list = os.listdir(path_file_3)
                for j in all_list:
                    number_ab += 1
                    name, suffix = j.rsplit(type)
                    name = str(number_ab) + type

                    old_name = path_file_3 + '\\' + j
                    new_name = path_file_new_3 + '\\' + name
                    shutil.copyfile(old_name, new_name)

            else:
                all_list = os.listdir(path_file_3)
                for j in all_list:
                    number_no += 1
                    name, suffix = j.rsplit(type)
                    name = str(number_no) + type

                    old_name = path_file_3 + '\\' + j
                    new_name = path_file_new_3 + '\\' + name
                    shutil.copyfile(old_name, new_name)

    print(number_ab+number_no)
    time_end = time.time()
    minute = (time_end - time_start) // 60
    second = (time_end - time_start) % 60
    print('\nTime cost', minute, 'min ', second, ' sec')