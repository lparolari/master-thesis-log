#!/bin/bash

project_dir=/home/lparolar/storage/vtg
mkdir -p "${project_dir}"

mkdir -p "${project_dir}/flickr30k_raw"
mkdir -p "${project_dir}/referit_raw"

cd "${project_dir}"

ln -s /home/drigoni/storage/datasets/flickr30k flickr30k
ln -s /home/drigoni/storage/datasets/refer refer
ln -s /home/drigoni/storage/datasets/YAGO YAGO

cd "${project_dir}/flickr30k_raw"

ln -s /home/drigoni/repository/VTKEL-solver/data/flickr30k_raw/out_bu out_bu
ln -s /home/drigoni/repository/VTKEL-solver/data/flickr30k_raw/out_ewiser out_ewiser

cd "${project_dir}/referit_raw"

ln -s /home/drigoni/repository/VTKEL-solver/data/referit_raw/out_bu out_bu
ln -s /home/drigoni/repository/VTKEL-solver/data/referit_raw/out_ewiser out_ewiser