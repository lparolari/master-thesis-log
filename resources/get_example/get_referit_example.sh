example_id="${example_id:-$1}"

if [ -z "${example_id}" ]; then
    echo "ERROR. No example id given"
    exit 1
fi

mkdir -p data/referit_raw/out_bu
mkdir -p data/referit_raw/out_ewiser
mkdir -p data/referit_raw/preprocessed

echo "${example_id}" >> data/referit_raw/train.txt
echo "${example_id}" >> data/referit_raw/test.txt
echo "${example_id}" >> data/referit_raw/val.txt

scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/referit_raw/yago_align.json data/referit_raw
scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/referit_raw/vocab_yago.json data/referit_raw
scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/referit_raw/vocab.json data/referit_raw

scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/referit_raw/out_bu/${example_id}.jpg.npz data/referit_raw/out_bu
scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/referit_raw/out_ewiser/${example_id}_{1,2,3,4,5}.json data/referit_raw/out_ewiser
# scp cluster.unipd:/home/lparolar/storage/VTKEL-solver/referit_raw/preprocessed/${example_id}_{0,1,2,3,4,img}.pickle data/referit_raw/preprocessed
