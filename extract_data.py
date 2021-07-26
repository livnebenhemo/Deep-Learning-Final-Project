import os
import cv2
import shutil
import sys
import numpy as np
import zipfile
import glob
import shutil
import time


def get_download_path():
    """Returns the default downloads path for linux or windows"""
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'downloads')


def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
path = './Data_7scenes'
temp_path = './temp'

try:
    os.mkdir(path)
except:
    shutil.rmtree(path)
    os.mkdir(path)
try:
    os.mkdir(temp_path)
except:
    shutil.rmtree(temp_path)
    os.mkdir(temp_path)

list_of_files = []
Download_path = get_download_path()
for file in scenes:
    list_of_files.extend(glob.glob(f'{Download_path}\\{file}.zip')) # * means all if need specific format then *.csv


for i, scene in enumerate(scenes):
    print(scene)
    try:
        with zipfile.ZipFile(f'{list_of_files[i]}', 'r') as zip_ref:
            zip_ref.extractall(f'{temp_path}')
        shutil.copytree(f'{temp_path}/{scene}', f'{temp_path}/{scene}', ignore=ig_f)
        time.sleep(5)
    except:
        pass

    for file in os.listdir(f'{temp_path}/{scene}'):
        try:
            if file.endswith(".zip"):
                with zipfile.ZipFile(f'{temp_path}/{scene}/{file}', 'r') as zip_ref:
                    zip_ref.extractall(f'{path}/{scene}')
            time.sleep(5)
        except:
            pass

    print(f'finished extracting zip file of {scene}')
    for i, dir in enumerate(os.listdir(f'{path}/{scene}')):
        for filename in os.listdir(f'{path}/{scene}/{dir}'):
            if filename.endswith(".png") and not filename.endswith(".depth.png"):
                image = cv2.imread(os.path.join(f'{path}/{scene}/{dir}/{filename}'), -1)
                smaller_dim = np.argmin([image.shape[0], image.shape[1]])
                if smaller_dim == 0:
                    new_image = cv2.resize(image, (int(image.shape[1] * (256 / image.shape[0])), 256))
                else:
                    new_image = cv2.resize(image, (256, int(image.shape[0] * (256 / image.shape[1]))), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f'{path}/{scene}/{dir}/{filename}', new_image)
            else:
                os.remove(f'{path}/{scene}/{dir}/{filename}')
            print(f'finished seq{i}')

    print(f'finished with {scene}')
    print(f'{len(scenes) - i - 1} scenes left')

try:
    shutil.rmtree(temp_path)
except:
    pass