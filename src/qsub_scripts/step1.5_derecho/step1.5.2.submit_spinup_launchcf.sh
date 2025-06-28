
cd /glade/work/guoqiang/CTSM_CAMELS/Calib_HH_MOASMO/submission

launch_cf -A NCGD0013 -q main -l select=1:ncpus=128 -l walltime=12:00:00 -l job_priority=economy -J 0-112 ./spinup_part1.txt
launch_cf -A NCGD0013 -q main -l select=1:ncpus=128 -l walltime=12:00:00 -l job_priority=economy -J 0-112 ./spinup_part2.txt
launch_cf -A NCGD0013 -q main -l select=1:ncpus=128 -l walltime=12:00:00 -l job_priority=economy -J 0-112 ./spinup_part3.txt
launch_cf -A NCGD0013 -q main -l select=1:ncpus=128 -l walltime=12:00:00 -l job_priority=economy -J 0-112 ./spinup_part4.txt
launch_cf -A NCGD0013 -q main -l select=1:ncpus=128 -l walltime=12:00:00 -l job_priority=economy -J 0-112 ./spinup_part5.txt
launch_cf -A NCGD0013 -q main -l select=1:ncpus=128 -l walltime=12:00:00 -l job_priority=economy -J 0-112 ./spinup_part6.txt

launch_cf -A NCGD0013 -q main -l select=1:ncpus=128 -l walltime=12:00:00 -l job_priority=preempt -J 0-10 ./spinup_rerun_failed.txt