
#module load conda
#conda activate npl-2022b-tgq

for i in {0..670}
do
  echo "processing basin $i"
  python create_CAMELS_config.py $i
done
