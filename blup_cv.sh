#Uses SNP set from nested_grid search (top or shuf)
#
#WARNING YOU MIGHT NEED TO CHNAGE "$snps"10ksnps_"$i"_in_4_out.txt to "$snps"10ksnps_"$i"_in_4_out.txt
pheno=$1 #eg FT10 or FT16
snps=$2 #top or shuf
for i in {1..4} ; do \
cut -d ' ' -f 1-2 train_raw_plink_"$i"_in_4_out.raw > /home/ciaran/arabadopsis/phenotypes/train_ids.txt ; \
cut -d ' ' -f 1-2 test_raw_plink_"$i"_in_4_out.raw > /home/ciaran/arabadopsis/phenotypes/test_ids.txt \
; \
/external_storage/eoin/GCTA_manual_install/gcta64 \
--bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids \
--extract "$snps"10ksnps_"$i"_in_4_out.txt \
--make-grm \
--thread-num 32 \
--out ./completed_big_matrix_binary_grm_"$pheno"_train_cv_"$i" \
--keep /home/ciaran/arabadopsis/phenotypes/train_ids.txt \
; \
/external_storage/eoin/GCTA_manual_install/gcta64 \
--reml \
--grm ./completed_big_matrix_binary_grm_"$pheno"_train_cv_"$i" \
--pheno /home/ciaran/arabadopsis/phenotypes/values_"$pheno" \
--reml-pred-rand \
--out ./"$pheno"_blup_solutions_train_cv_"$i" \
--keep /home/ciaran/arabadopsis/phenotypes/train_ids.txt \
; \
/external_storage/eoin/GCTA_manual_install/gcta64 \
--bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids \
--blup-snp ./"$pheno"_blup_solutions_train_cv_"$i".indi.blp \
--extract "$snps"10ksnps_"$i"_in_4_out.txt \
--keep /home/ciaran/arabadopsis/phenotypes/train_ids.txt \
--out ./"$pheno"_gblup_snp_FX_train_only_grm_cv_"$i" \
; \
plink1.9 \
--bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids \
--keep /home/ciaran/arabadopsis/phenotypes/test_ids.txt \
--out ./"$pheno"_gblup_train_only_grm_cv_"$i"_test \
--pheno /home/ciaran/arabadopsis/phenotypes/values_"$pheno" \
--score ./"$pheno"_gblup_snp_FX_train_only_grm_cv_"$i".snp.blp 1 2 3 \
; \
python /external_storage/ciaran/machine_learning2/r2_score.py ./"$pheno"_gblup_train_only_grm_cv_"$i"_test.profile "$i" >> "$pheno"_gblup_train_only_grm_cv_test.prscores \
; done
