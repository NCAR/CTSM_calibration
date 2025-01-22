
# module load conda
# conda activate npl-2022b-tgq

level="level1"
for i in {0..626}
do
  echo "processing basin $i"
  python create_CAMELS_config_Ostrich.py $i $level
done


level="level2"
for i in {0..39}
do
  echo "processing basin $i"
  python create_CAMELS_config_Ostrich.py $i $level
done

level="level3"
for i in {0..3}
do
  echo "processing basin $i"
  python create_CAMELS_config_Ostrich.py $i $level
done

