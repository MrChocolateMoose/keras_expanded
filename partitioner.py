from glob import glob
import zipfile
import random
import shutil
import os.path
import math

def stratified_partition(all_data_dir, data_dir, named_partitions):
    all_data_subdirs = glob(all_data_dir + "/*/")

    stratified_partition_subdirs(all_data_dir, all_data_subdirs, data_dir, named_partitions)

def stratified_partition_subdirs(all_data_dir, all_data_label_dirs, new_data_dir, named_partitions, new_label_dir = None, use_src_label_dir_as_fname_prefix = False):

    # make paths to new partitioned label dirs
    for cur_label_dir in all_data_label_dirs:
        for partition_name, partition_pct in named_partitions.items():

            if new_label_dir == None:
                label_dir = os.path.basename(cur_label_dir.rstrip("/"))
            else:
                label_dir = new_label_dir

            new_data_label_dir = os.path.join(new_data_dir, partition_name, label_dir)

            if not os.path.exists(new_data_label_dir):
                os.makedirs(new_data_label_dir)

    # move files from label dirs to new partitioned paths
    for cur_label_dir in all_data_label_dirs:

        cur_label_dir_files = glob(cur_label_dir + "/*")

        # count all files in cur label dir
        cur_label_dir_files_len = len(cur_label_dir_files)

        for partition_name, partition_pct in named_partitions.items():
            # get remaining files in label dir and len
            cur_remaining_label_dir_files = glob(cur_label_dir + "/*")
            cur_remaining_label_dir_files_len = len(cur_remaining_label_dir_files)

            files_partition_size = min(int(math.ceil(cur_label_dir_files_len * partition_pct)),
                                       cur_remaining_label_dir_files_len)

            # sample from remaining
            partitioned_label_dir_files = random.sample(cur_remaining_label_dir_files, files_partition_size)

            for file_to_partition in partitioned_label_dir_files:

                source = file_to_partition

                if new_label_dir == None:
                    label_dir = os.path.basename(os.path.dirname(file_to_partition))
                else:
                    label_dir = new_label_dir

                if use_src_label_dir_as_fname_prefix == True:
                    fname =  os.path.basename(os.path.dirname(file_to_partition)) + "_" + os.path.basename(file_to_partition)
                else:
                    fname = os.path.basename(file_to_partition)

                dest = os.path.join(new_data_dir, partition_name, label_dir, fname)

                shutil.move(source, dest)

    #TODO: perform destructive delete of original dir? How to prevent unintended consequences?
    #os.removedirs(all_data_dir)

def extract_data(data_dir_name = "data", force_remove = False):

    abs_data_dir = os.path.abspath(data_dir_name)

    while len(glob(abs_data_dir + "/*/")) != 0:
        msg = "Delete all data in folder: %s?" % abs_data_dir

        if not force_remove:
            force_remove = input("%s (y/N) " % msg).lower() == 'y'

        if force_remove:
            shutil.rmtree(abs_data_dir)

    with zipfile.ZipFile(data_dir_name + ".zip") as zf:
        zf.extractall(path=abs_data_dir)

