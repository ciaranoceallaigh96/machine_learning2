#!/bin/bash

phenofile='/home/ciaran/arabadopsis/phenotypes/values_FT10.8424.80.del'

#phenotype file is first argument e.g /home/ciaran/arabadopsis/phenotypes/values_FT16.8424.80.del
#cleanup
#rm test_raw_plink* ; rm train_raw_plink*
echo "$1"
echo "$2"
echo "$3"
#conduct GWAS
plink2 --glm --mac 20 --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids --out nested_cv_gwas_out_"$1"_in_"$2"_"$3" --keep name_vector_train.txt --pheno $phenofile
echo "red"
cat header.txt <(sort -g -k 12,12 nested_cv_gwas_out_"$1"_in_"$2"_"$3".FT16.glm.linear | awk '{if ($12 != "NA") print}' | tail -n +2) > gwas_results_"$1"_in_"$2"_"$3".gsorted #formatting
echo "blue"
#clump
plink1.9 --prune --pheno $phenofile --bfile /home/alexg/hopefully_final/completed_big_matrix_binary_new_snps_ids --clump-kb 250 --clump-p1 1 --clump-p2 1 --clump-r2 0.1 --clump gwas_results_"$1"_in_"$2"_"$3".gsorted --out gwas_results_clumped_"$1"_in_"$2"_"$3"
echo "yellow"
head -n 10000 gwas_results_clumped_"$1"_in_"$2"_"$3".clumped | awk '{print $3}'  > top10ksnps_"$1"_in_"$2"_"$3".txt
#extract top snps
plink1.9 --prune --pheno $phenofile --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids --keep name_vector_train.txt --extract top10ksnps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_"$1"_in_"$2"_"$3"

plink1.9 --prune --pheno $phenofile --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids --keep name_vector_test.txt --extract top10ksnps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_"$1"_in_"$2"_"$3"
echo "purple"


#rm name_vector_train.txt ; rm name_vector_test.txt
