#!/bin/bash

project_dir=/home/lparolar/storage/VTKEL-solver
mkdir -p "${project_dir}"

cd "${project_dir}"

ln -s /home/drigoni/storage/datasets/flickr30k flickr30k
ln -s /home/drigoni/storage/datasets/refer refer
ln -s /home/drigoni/storage/datasets/YAGO YAGO

mkdir -p "${project_dir}/flickr30k_raw"
cd "${project_dir}/flickr30k_raw"

ln -s /home/drigoni/repository/VTKEL-solver/data/flickr30k_raw/out_bu
ln -s /home/drigoni/repository/VTKEL-solver/data/flickr30k_raw/out_ewiser
ln -s /home/drigoni/repository/VTKEL-solver/data/flickr30k_raw/out_ewiser_queries
ln -s /home/drigoni/repository/VTKEL-solver/data/flickr30k_raw/yago_align.json
