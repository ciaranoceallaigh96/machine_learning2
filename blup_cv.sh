pheno=$1

for i in {1..1} ; do \
cut -d ' ' -f 1-2 train_raw_plink_shuf_"$i"_in_4_out.raw > /home/ciaran/arabadopsis/phenotypes/train_ids.txt ; \
cut -d ' ' -f 1-2 test_raw_plink_shuf_"$i"_in_4_out.raw > /home/ciaran/arabadopsis/phenotypes/test_ids.txt \
; \
\ #Construct GRM \
/external_storage/eoin/GCTA_manual_install/gcta64 \
--bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids \
--extract shuf_10000_snps_1_in_4_out.txt \
--make-grm \
--thread-num 32 \
--out /external_storage/ciaran/greml/completed_big_matrix_binary_grm_"$pheno"_train_cv_"$i" \
--keep /home/ciaran/arabadopsis/phenotypes/train_ids.txt \
#--remove /home/ciaran/arabadopsis/phenotypes/values_"$pheno".nofailids.txt \
; \
\ #To obtain BLUP solutions for the genetic values of individuals \
--reml \
--grm /external_storage/ciaran/greml/completed_big_matrix_binary_grm_"$pheno"_train_cv_"$i" \
--pheno /home/ciaran/arabadopsis/phenotypes/values_"$pheno" \
--reml-pred-rand \
--out /external_storage/ciaran/greml/"$pheno"_blup_solutions_train_cv_"$i" \
#--remove /home/ciaran/arabadopsis/completed_big_matrix_binary_new_snps_fail_ids \
--keep /home/ciaran/arabadopsis/phenotypes/train_ids.txt \
; \
#To obtain BLUP solutions for the SNP effects \
--bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids \
--blup-snp /external_storage/ciaran/greml/"$pheno"_blup_solutions_train_cv_"$i".indi.blp \
--extract shuf_10000_snps_1_in_4_out.txt \
--keep /home/ciaran/arabadopsis/phenotypes/train_ids.txt \
--out /external_storage/ciaran/greml/"$pheno"_gblup_snp_FX_train_only_grm_cv_"$i" \
; \
#To compute the polygenic risk score in an independent sample \
--bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids \
--keep /home/ciaran/arabadopsis/phenotypes/test_ids.txt \
--out /external_storage/ciaran/greml/"$pheno"_gblup_train_only_grm_cv_"$i"_test \
--pheno /home/ciaran/arabadopsis/phenotypes/values_"$pheno" \
#--remove /home/ciaran/arabadopsis/completed_big_matrix_binary_new_snps_fail_ids \
--score /external_storage/ciaran/greml/"$pheno"_gblup_snp_FX_train_only_grm_cv_"$i".snp.blp 1 2 3 \
; done
