#!/bin/bash
phenofile='/home/ciaran/arabadopsis/phenotypes/values_FT16.8424.80.del'
echo "liath"
echo "$1"
echo "$2"
echo "$3"

#phenotype file is first argument e.g /home/ciaran/arabadopsis/phenotypes/values_FT16.8424.80.del
#cleanup
#rm test_raw_plink* ; rm train_raw_plink*
echo "dearg"
#choose shuf set
shuf /external_storage/ciaran/greml/indep_snps_full_dataset_new_snp_ids.prune.in | head -n 10000 > shuf10ksnps_"$1"_in_"$2"_"$3".txt 
echo "gorm"
#extract shuf snps
plink1.9 --pheno $phenofile --prune --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids --keep name_vector_train.txt --extract shuf10ksnps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_shuf_"$1"_in_"$2"_"$3"

plink1.9 --pheno $phenofile --prune --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids --keep name_vector_test.txt --extract shuf10ksnps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_shuf_"$1"_in_"$2"_"$3"
echo "corcra"


rm name_vector_train.txt ; rm name_vector_test.txt
