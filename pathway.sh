python machine_learning2/cv_saver.py 5 /home/ciaran/completed_big_matrix_binary_new_snps_ids.fam > cv_saver.log


for i in {1..5}; do plink2 --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids --indep-pairwise 250 5 0.05 --mac 6 --out /external_s
torage/ciaran/greml/indep_snps_full_dataset_new_snp_ids_cv_"$i" --keep ~/arabadopsis/2021/train_split_cv_"$i".txt ; done

for i in {1..5}; do /external_storage/eoin/GCTA_manual_install/gcta64 --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids \
--extract /external_storage/ciaran/greml/indep_snps_full_dataset_new_snp_ids_cv_"$i".prune.in --keep  ~/arabadopsis/2021/train_split_cv_"$i".txt --make-grm --thread-num 32 \
--remove /home/ciaran/arabadopsis/completed_big_matrix_binary_new_snps_fail_ids --out /external_storage/ciaran/greml/completed_big_matrix_binary_grm_cv_sep_"$i" ; done


for i in {1..5}; do /external_storage/eoin/GCTA_manual_install/gcta64  --reml --grm /external_storage/ciaran/greml/completed_big_matrix_binary_grm_cv_sep_"$i" /
--reml-pred-rand --pheno /home/ciaran/arabadopsis/phenotypes/values_FT16 --out /external_storage/ciaran/greml/blup_solutions_cv_"$i"_FT16 /
--keep  ~/arabadopsis/2021/train_split_cv_"$i".txt --remove /home/ciaran/arabadopsis/completed_big_matrix_binary_new_snps_fail_ids; done

for i in {1..5}; do /external_storage/eoin/GCTA_manual_install/gcta64 --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids /
--blup-snp /external_storage/ciaran/greml/blup_solutions_cv_"$i"_FT16.indi.blp --out snp_blup_solutions_cv_"$i"_FT16 ; done
