#!/bin/bash
#SBATCH --time=09:35:00
#SBATCH --mem=5G
#SBATCH --mail-type=END
#SBATCH --mail-user=oceallc@tcd.ie
#SBATCH --cpus-per-task 32
#SBATCH --chdir /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest
#SBATCH --open-mode=append 

#/hpc/local/CentOS7/hers_en/software/plink-1.90/plink --allow-no-sex --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_plink --extract /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/mouse/mouse_startle_top_resids/top_20000_snps_1_in_1_in.txt --keep /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/mouse/mouse_startle_top_resids/name_vector_test.txt --out /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/mouse/mouse_startle_top_resids/test_raw_plink_top_1_in_1_in2 --pheno /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/phenotypes/startle.resids.txt --prune --recode A
#/hpc/local/CentOS7/hers_en/software/plink-1.90/plink --allow-no-sex --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_plink --extract /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/mouse/mouse_startle_top_resids/top_20000_snps_1_in_1_in.txt --keep /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/mouse/mouse_startle_top_resids/name_vector_train.txt --out /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/nest/mouse/mouse_startle_top_resids/train_raw_plink_top_1_in_1_in2 --pheno /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/phenotypes/startle.resids.txt --prune --recode A

# sbatch --error ml_grid_nested12.err --output ml_grid_nested12.log  ml_grid_nested.sh 5 nahnah /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/stratum/all_all_top_clumped_5000_raw_first1000.raw no_clump als_nest_top 1004
source /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/envciaran2/bin/activate
/hpc/local/CentOS7/common/lang/python/3.6.1/bin/python3.6 -u /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/machine_learning2/learning_rate_cycler.py $1 $2 ##### > to log.txt
