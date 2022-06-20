import functools
import os.path
import pathlib
import urllib.parse
from tensorflow.keras.utils import get_file

def download_data(url, data_dir, filename='', extract=False):
    if not filename:
        filename = os.path.basename(urllib.parse.urlparse(url).path)
    file_path = pathlib.Path(data_dir, filename)
    data_dir = pathlib.Path(data_dir)

    download = lambda : get_file( filename,
                                        origin=url,
                                        extract=extract,
                                        cache_dir=str(data_dir.parents[0]),
                                        cache_subdir=str(data_dir)
                                )

    if not data_dir.exists() and not file_path.exists():
        print("Data directory does not exist. Creating directory " + str(data_dir))
        download()
    elif not file_path.exists():
        opt = ''
        while opt == '':
            opt = input("Directory " + str(data_dir) + " already exists. Do you wish to continue with the download process? [Y/N]\n")
            l_opt = opt.lower()
            if l_opt == 'y':
                download()
            elif l_opt == 'n':
                print("Download cancelled.")
                break
            else:
                print("Invalid input: " + opt)
                opt = ''
    else:
        print("File already exists at " + str(file_path))
    
def parse_list(string_input, type=str, delimiter=',', ignore=''):
    return list(map(lambda x: type(x.translate(str.maketrans('','', ignore))), string_input.split(delimiter)))

def next_power_of_two(x):
    return 1 if x == 0 else 2**(int(x) - 1).bit_length()

def compose(func_list):
    return functools.reduce(lambda f1,f2: lambda x: f1(f2(x)), func_list, lambda x:x)

def map_dict(func, dictionary):
    return {k : func(k, v) for (k, v) in dictionary.items()}

def path_exists(str):
    return pathlib.Path(str).exists()

def assert_exists(error_msg_func=(lambda n,o: "Required argument " + n + " not provided or invalid. Got '" + str(o) +"'"), **kwargs):
    for (n, o) in kwargs.items():
        if not o:
            raise Exception(error_msg_func(n, o))