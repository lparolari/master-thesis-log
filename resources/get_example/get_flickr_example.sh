example_id="${example_id:-$1}"

if [ -z "${example_id}" ]; then
    echo "ERROR. No example id given"
    exit 1
fi

mkdir -p data/flickr30k/flickr30k_entities/Annotations
mkdir -p data/flickr30k/flickr30k_entities/Sentences
mkdir -p data/flickr30k/flickr30k_images
mkdir -p data/flickr30k_raw/preprocessed
mkdir -p data/flickr30k_raw/out_bu
mkdir -p data/flickr30k_raw/out_ewiser
mkdir -p data/flickr30k_raw/out_ewiser_queries

# flickr30k
scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/flickr30k/flickr30k_entities/Annotations/${example_id}.xml data/flickr30k/flickr30k_entities/Annotations
scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/flickr30k/flickr30k_entities/Sentences/${example_id}.txt data/flickr30k/flickr30k_entities/Sentences
scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/flickr30k/flickr30k_images/${example_id}.jpg data/flickr30k/flickr30k_images

echo "${example_id}" > data/flickr30k/flickr30k_entities/train.txt
echo "${example_id}" > data/flickr30k/flickr30k_entities/test.txt
echo "${example_id}" > data/flickr30k/flickr30k_entities/val.txt

# flickr30k raw
scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/flickr30k_raw/yago_align.json data/flickr30k_raw
scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/flickr30k_raw/vocab_yago.json data/flickr30k_raw
scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/flickr30k_raw/vocab.json data/flickr30k_raw

scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/flickr30k_raw/preprocessed/${example_id}_{0,1,2,3,4,img}.pickle data/flickr30k_raw/preprocessed

scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/flickr30k_raw/out_bu/${example_id}.jpg.npz data/flickr30k_raw/out_bu
scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/flickr30k_raw/out_ewiser/${example_id}.txt.json data/flickr30k_raw/out_ewiser
scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/flickr30k_raw/out_ewiser_queries/${example_id}.txt.json data/flickr30k_raw/out_ewiser_queries
