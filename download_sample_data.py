import remotezip as rz
import os
import pathlib
import tqdm
import collections
import random


def list_files_from_zip(path :str) -> list:
    """

    :param path:
    :return:
    """
    files = []
    with rz.RemoteZip(path) as zip:
        for zip_info in zip.infolist():
            files.append(zip_info.filename)
        return files

def get_class(fname):
  return fname.split('_')[-3]

def get_files_per_class(files):
  files_for_class = collections.defaultdict(list)
  for fname in files:
    class_name = get_class(fname)
    files_for_class[class_name].append(fname)
  return files_for_class

def select_subset_of_classes(files_for_class, files_per_class, classes):
  files_subset = dict()
  for class_name in classes:
    class_file = files_for_class[class_name]
    files_subset[class_name] = class_file[: files_per_class]
  return files_subset


def download_from_zip(zip_url, to_dir, filename):
    with rz.RemoteZip(zip_url) as zip:
        for fn in tqdm.tqdm(filename):
            class_name = get_class(fn)
            zip.extract(fn, str(to_dir / class_name))
            unzipped_file = to_dir / class_name / fn

            fn = pathlib.Path(fn).parts[-1]
            out_file = to_dir / class_name / fn
            unzipped_file.rename(out_file)


def split_class_list(files_for_class, count):
  split_files = []
  remainder = {}
  for cls in files_for_class:
    split_files.extend(files_for_class[cls][:count])
    remainder[cls] = files_for_class[cls][count:]
  return split_files, remainder


def download_data(zip_url, num_classes, splits, download_dir):
    files = list_files_from_zip(zip_url)

    for f in files:
        path = os.path.normpath(f)
        tokens = path.split(os.sep)
        if len(tokens) <= 2:
            files.remove(f)

    files_for_class = get_files_per_class(files)
    classes = list(files_for_class.keys())[:num_classes]

    for cls in classes:
        random.shuffle(files_for_class[cls])

    dirs = {}
    for split_name, split_count in splits.items():
        print(split_name, ":")
        split_dir = download_dir / split_name
        split_files, files_for_class = split_class_list(files_for_class,
                                                        split_count)
        download_from_zip(zip_url, split_dir, split_files)
        dirs[split_name] = split_dir
    return dirs

if __name__=="__main__":
    url = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'
    os.makedirs("sample_code/data/UCF101_subset/", exist_ok=True)

    download_dir = pathlib.Path('sample_code/data/UCF101_subset/')
    num_class = 10
    subset_paths = download_data(url,
                                 num_classes = num_class,
                                 splits = {"train": 30, "val": 10, "test": 10},
                                 download_dir = download_dir)