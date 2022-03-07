#!/bin/bash
#SBATCH --time=00:35:00
#SBATCH --mem=10G
#SBATCH --mail-type=END
#SBATCH --mail-user=oceallc@tcd.ie
#SBATCH --cpus-per-task 32
#SBATCH --chdir /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest
#SBATCH --open-mode=append 

# sbatch --error ml_grid_nested12.err --output ml_grid_nested12.log  ml_grid_nested.sh 5 nahnah /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/stratum/all_all_top_clumped_5000_raw_first1000.raw no_clump als_nest_top 1004
source /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/envciaran2/bin/activate
/hpc/local/CentOS7/common/lang/python/3.6.1/bin/python3.6 -u /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/machine_learning2/cv_grid_all_ml_expanded.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ##### > to log.txt
