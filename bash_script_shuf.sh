#!/bin/bash

phenofile='/home/ciaran/arabadopsis/phenotypes/values_FT16.8424.80.del'

#phenotype file is first argument e.g /home/ciaran/arabadopsis/phenotypes/values_FT16.8424.80.del
#cleanup
rm test_raw_plink* ; rm train_raw_plink*

#extract shuf snps
plink1.9 --pheno $phenofile --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids --keep name_vector_train.txt --extract shuf10ksnps.txt --recode A --out train_raw_plink_shuf

plink1.9 --pheno $phenofile --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids --keep name_vector_test.txt --extract shuf10ksnps.txt --recode A --out test_raw_plink_shuf
echo "purple"


rm name_vector_train.txt ; rm name_vector_test.txt
rm nested_cv_gwas_to_delete.*
rm gwas_results*

