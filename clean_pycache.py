'''
This script is used to clean the __pycache__ folders in the current directory.
'''

import os
import shutil


# Define the directory to start searching from
dir_list = []
for subdir in os.listdir():
    if not '.' in subdir and not 'Data' in subdir:
        dir_list.append(subdir)


# Walk through all directories and files in the repository
for dir2check in dir_list:
    for dirpath, dirnames, filenames in os.walk(dir2check):
        # If a __pycache__ directory is found, remove it
        if '__pycache__' in dirnames:
            cache_path = os.path.join(dirpath, '__pycache__')
            shutil.rmtree(cache_path)
            print(f"Removed {cache_path}")


# Clear the __pycache__ in the current directory
if '__pycache__' in os.listdir():
    shutil.rmtree('__pycache__')
    print(f"Removed __pycache__ in the current directory")