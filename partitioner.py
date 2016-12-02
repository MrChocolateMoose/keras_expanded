from glob import glob
import random
import shutil
import os.path
import math

def stratified_partition(all_data_dir, data_dir, named_partitions):
    all_data_subdirs = glob(all_data_dir + "/*/")

    stratified_partition_subdirs(all_data_dir, all_data_subdirs, data_dir, named_partitions)

def stratified_partition_subdirs(all_data_dir, all_data_subdirs, data_dir, named_partitions):

    for cur_data_subdir in all_data_subdirs:
        for name, partition_pct in named_partitions.items():

            partition_data_subdir = cur_data_subdir.replace(all_data_dir, os.path.join(data_dir, name))

            if not os.path.exists(partition_data_subdir):
                os.makedirs(partition_data_subdir)

    for cur_data_subdir in all_data_subdirs:

        cur_subdir_files = glob(cur_data_subdir + "/*")

        # count all files in subdir
        cur_subdir_files_len = len(cur_subdir_files)

        for name, partition_pct in named_partitions.items():
            # get remaining files in subdir and len
            cur_remaining_subdir_files = glob(cur_data_subdir + "/*")
            cur_remaining_subdir_files_len = len(cur_remaining_subdir_files)

            files_partition_size = min(int(math.ceil(cur_subdir_files_len * partition_pct)),
                                       cur_remaining_subdir_files_len)

            # sample from remaining
            partitioned_subdir_files = random.sample(cur_remaining_subdir_files, files_partition_size)

            for file_to_partition in partitioned_subdir_files:
                source = file_to_partition
                dest = file_to_partition.replace(all_data_dir, os.path.join(data_dir, name))

                shutil.move(source, dest)

    #TODO: perform destructive delete of original dir? How to prevent unintended consequences?
    #os.removedirs(all_data_dir)
