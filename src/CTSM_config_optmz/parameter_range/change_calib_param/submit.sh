#PBS -N MOAcalib
#PBS -q develop
#PBS -l select=1:ncpus=56:mem=150gb
#PBS -l walltime=6:00:00
#PBS -A P08010000


module load conda
conda activate npl-2024a-tgq


b=(30 40 41 42 43 44 45 94 107 129 134 135 136 137 138 139 140 141 142 143 144 145 147 148 149 207 208 218 243 244 255 258 259 260 261 270 271 272 273 274 275 277 279 281 335 336 338 403 404 405 406 412 414 426 428 434)

parallel python create_case_and_run.py ::: "${b[@]}"