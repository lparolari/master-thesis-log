#!/bin/bash

# Mount the storage/vtg directory to local directory through sshfs
sshfs -o follow_symlinks cluster.unipd:/home/lparolar/storage/vtg/ /mnt/cluster/vtg