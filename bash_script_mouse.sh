#!/bin/bash

#phenofile='/home/ciaran/arabadopsis/phenotypes/values_FT16.8424.80.del'
#phenofile='/home/ciaran/arabadopsis/phenotypes/values_FT10.8424.dup.del'
#pheno='FT10'
#phenotype file is first argument e.g /home/ciaran/arabadopsis/phenotypes/values_FT16.8424.80.del
#cleanup
#rm test_raw_plink* ; rm train_raw_plink*
echo "$1"
echo "$2"
echo "$3"
#conduct GWAS
#plink2 --glm --mac 20 --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids --out nested_cv_gwas_out_"$1"_in_"$2"_"$3" --keep name_vector_train.txt --pheno $phenofile
#echo "red"
#cat header.txt <(sort -g -k 12,12 nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.linear | awk '{if ($12 != "NA") print}' | tail -n +2) > gwas_results_"$1"_in_"$2"_"$3".gsorted #formatting
echo "blue"
#clump
#plink1.9 --prune --pheno $phenofile --bfile /home/alexg/hopefully_final/completed_big_matrix_binary_new_snps_ids --clump-kb 250 --clump-p1 1 --clump-p2 1 --clump-r2 0.1 --clump gwas_results_"$1"_in_"$2"_"$3".gsorted --out gwas_results_clumped_"$1"_in_"$2"_"$3"
#echo "yellow"
#head -n 10000 gwas_results_clumped_"$1"_in_"$2"_"$3".clumped | awk '{print $3}'  > top10ksnps_"$1"_in_"$2"_"$3".txt
#extract top snps
#plink1.9 --prune --pheno $phenofile --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids --keep name_vector_train.txt --extract top10ksnps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_"$1"_in_"$2"_"$3"

#plink1.9 --prune --pheno $phenofile --bfile /home/ciaran/completed_big_matrix_binary_new_snps_ids --keep name_vector_test.txt --extract top10ksnps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_"$1"_in_"$2"_"$3"
echo "purple"


#rm name_vector_train.txt ; rm name_vector_test.txt

if [ "$6" == "shuf" ] 
then

#choose shuf set
        pheno=$(awk '{print $3}' $4 | head -n 1)
        echo "Pheno is $pheno"
        phenofile="$4"
	#plink2 --bfile /external_storage/ciaran/mouse/mouse_plink --indep-pairwise 5 5 0.5 --out /external_storage/ciaran/greml/mouse_indep_snps #mac 6 doesnt do anything
	shuf /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_indep_snps.prune.in | head -n $5 > shuf_"$5"_snps_"$1"_in_"$2"_"$3".txt

	/hpc/local/CentOS7/hers_en/software/plink-1.90/plink --pheno $phenofile --prune --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_plink   --keep name_vector_train.txt --extract shuf_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_shuf_"$1"_in_"$2"_"$3"

	/hpc/local/CentOS7/hers_en/software/plink-1.90/plink  --pheno $phenofile --prune --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_plink   --keep name_vector_test.txt --extract shuf_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_shuf_"$1"_in_"$2"_"$3"

	mv name_vector_train.txt name_vector_train_"$1"_in_"$2"_"$3".txt ; mv name_vector_test.txt name_vector_test_"$1"_in_"$2"_"$3".txt

fi

if [ "$6" == "top_covar" ]
then
        pheno=$(awk '{print $3}' $4 | head -n 1)
        echo "Pheno is $pheno"
        phenofile="$4"
        #sed "s/'/ /g" name_vector_train.txt | awk '{print $2, $3}' > name_vector_train2.txt; mv name_vector_train2.txt name_vector_train.txt
        #sed "s/'/ /g" name_vector_test.txt | awk '{print $2, $3}' > name_vector_test2.txt; mv name_vector_test2.txt name_vector_test.txt
#choose top set
        /hpc/hers_en/rmclaughlin/ross/programs/plink2/plink2 --covar covar_file_"$pheno".txt --out nested_cv_gwas_out_"$1"_in_"$2"_"$3" --allow-no-sex --glm --mac 20 --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_plink --keep name_vector_train.txt --pheno $phenofile

        if  test -f nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.linear ; then cat /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/machine_learning2/header.txt <(sort -g -k 12,12 nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.linear | awk '{if ($12 != "NA") print}' | tail -n +2) > gwas_results_"$1"_in_"$2"_"$3".gsorted ; fi #formatting
        if test -f nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.logistic ; then cat /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/machine_learning2/header.txt <(sort -g -k 12,12 nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.logistic | awk '{if ($12 != "NA") print}' | tail -n +2) > gwas_results_"$1"_in_"$2"_"$3".gsorted ; fi #formatting
        #awk '{if ($12 <= 0.01) print}' gwas_results_"$1"_in_"$2"_"$3".gsorted > gwas_results_"$1"_in_"$2"_"$3".gsorted.001
        #echo "WARNING FILTERING RESULTS OF GWAS KESS THAN 0.01"
        #cat header.txt gwas_results_"$1"_in_"$2"_"$3".gsorted.001 > gwas_results_"$1"_in_"$2"_"$3".gsorted.001.filter
        /hpc/local/CentOS7/hers_en/software/plink-1.90/plink --prune --allow-no-sex --pheno $phenofile --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse/mouse_plink  --clump-kb 5 --clump-p1 0.2 --clump-p2 0.5 --clump-r2 0.5 --clump gwas_results_"$1"_in_"$2"_"$3".gsorted --out gwas_results_clumped_"$1"_in_"$2"_"$3"
        head -n $5 gwas_results_clumped_"$1"_in_"$2"_"$3".clumped | awk '{print $3}'  > top_"$5"_snps_"$1"_in_"$2"_"$3".txt

        /hpc/local/CentOS7/hers_en/software/plink-1.90/plink --prune --allow-no-sex --pheno $phenofile --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_plink --keep name_vector_train.txt --extract top_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_top_"$1"_in_"$2"_"$3"

        /hpc/local/CentOS7/hers_en/software/plink-1.90/plink --prune --allow-no-sex --pheno $phenofile --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_plink --keep name_vector_test.txt --extract top_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_top_"$1"_in_"$2"_"$3"


fi


if [ "$6" == "top" ]
then
        pheno=$(awk '{print $3}' $4 | head -n 1)
        echo "Pheno is $pheno"
        phenofile="$4"
        #sed "s/'/ /g" name_vector_train.txt | awk '{print $2, $3}' > name_vector_train2.txt; mv name_vector_train2.txt name_vector_train.txt
        #sed "s/'/ /g" name_vector_test.txt | awk '{print $2, $3}' > name_vector_test2.txt; mv name_vector_test2.txt name_vector_test.txt
#choose top set
        /hpc/hers_en/rmclaughlin/ross/programs/plink2/plink2 --out nested_cv_gwas_out_"$1"_in_"$2"_"$3" --allow-no-sex --glm --mac 20 --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_plink --keep name_vector_train.txt --pheno $phenofile

        if  test -f nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.linear ; then cat /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/machine_learning2/header.txt <(sort -g -k 12,12 nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.linear | awk '{if ($12 != "NA") print}' | tail -n +2) > gwas_results_"$1"_in_"$2"_"$3".gsorted ; fi #formatting
        if test -f nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.logistic ; then cat /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/machine_learning2/header.txt <(sort -g -k 12,12 nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.logistic | awk '{if ($12 != "NA") print}' | tail -n +2) > gwas_results_"$1"_in_"$2"_"$3".gsorted ; fi #formatting
        #awk '{if ($12 <= 0.01) print}' gwas_results_"$1"_in_"$2"_"$3".gsorted > gwas_results_"$1"_in_"$2"_"$3".gsorted.001
        #echo "WARNING FILTERING RESULTS OF GWAS KESS THAN 0.01"
        #cat header.txt gwas_results_"$1"_in_"$2"_"$3".gsorted.001 > gwas_results_"$1"_in_"$2"_"$3".gsorted.001.filter
        ###OLD --clump-kb 5 --clump-p1 0.2 --clump-p2 0.5 --clump-r2 0.5
        /hpc/local/CentOS7/hers_en/software/plink-1.90/plink --prune --allow-no-sex --pheno $phenofile --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_plink  --clump-kb 15 --clump-p1 0.2 --clump-p2 0.5 --clump-r2 0.5 --clump gwas_results_"$1"_in_"$2"_"$3".gsorted --out gwas_results_clumped_"$1"_in_"$2"_"$3"
        head -n $5 gwas_results_clumped_"$1"_in_"$2"_"$3".clumped | awk '{print $3}'  > top_"$5"_snps_"$1"_in_"$2"_"$3".txt

        /hpc/local/CentOS7/hers_en/software/plink-1.90/plink --prune --allow-no-sex --pheno $phenofile --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_plink --keep name_vector_train.txt --extract top_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_top_"$1"_in_"$2"_"$3"

        /hpc/local/CentOS7/hers_en/software/plink-1.90/plink --prune --allow-no-sex --pheno $phenofile --bfile /hpc/hers_en/rmclaughlin/ciaran/keras_tryout/mouse/mouse_plink --keep name_vector_test.txt --extract top_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_top_"$1"_in_"$2"_"$3"


fi





if [ "$6" == "no_clump" ]
then
        pheno=$(awk '{print $3}' $4 | head -n 1)
        echo "Pheno is $pheno"
        phenofile="$4"
        #grep -v -n "' '" name_vector_train.txt | cut -d ':' -f 1 > line_file.txt #which lines have been spread over two lines
        #grep -v -n "' '" name_vector_test.txt | cut -d ':' -f 1 > line_file.txt2
	#for i in $(awk 'NR % 2 == 1' line_file.txt); do sed "$i N;s/\n//" name_vector_train.txt > name_vector_train2.txt; mv name_vector_train2.txt name_vector_train.txt ; done #for the first line in every pair replace the Nth newline with nothing ###weird bug of saving out saves long names over two lines #bug fix 
        #for i in $(awk 'NR % 2 == 1' line_file.txt2); do sed "$i N;s/\n//" name_vector_test.txt > name_vector_test2.txt; mv name_vector_test2.txt name_vector_test.txt ; done
        #sed "s/'/ /g" name_vector_train.txt | awk '{print $2, $3}' > name_vector_train2.txt; mv name_vector_train2.txt name_vector_train.txt
        #sed "s/'/ /g" name_vector_test.txt | awk '{print $2, $3}' > name_vector_test2.txt; mv name_vector_test2.txt name_vector_test.txt
#choose top set
        plink2 --out nested_cv_gwas_out_"$1"_in_"$2"_"$3" --allow-no-sex --glm --mac 20 --bfile /external_storage/ciaran/mouse/mouse_plink  --keep name_vector_train.txt --pheno $phenofile

        cat /external_storage/ciaran/machine_learning2/header.txt <(sort -g -k 12,12 nested_cv_gwas_out_"$1"_in_"$2"_"$3"."$pheno".glm.linear | awk '{if ($12 != "NA") print}' | tail -n +2) > gwas_results_"$1"_in_"$2"_"$3".gsorted #formatting

        #awk '{if ($12 <= 0.01) print}' gwas_results_"$1"_in_"$2"_"$3".gsorted > gwas_results_"$1"_in_"$2"_"$3".gsorted.001
        #echo "WARNING FILTERING RESULTS OF GWAS KESS THAN 0.01"
        #cat header.txt gwas_results_"$1"_in_"$2"_"$3".gsorted.001 > gwas_results_"$1"_in_"$2"_"$3".gsorted.001.filter

        head -n $5 gwas_results_"$1"_in_"$2"_"$3".gsorted | awk '{if ($3 != "") print $3}'  > top_"$5"_snps_"$1"_in_"$2"_"$3".txt #has header

        plink1.9 --1 --prune --allow-no-sex --pheno $phenofile --bfile /external_storage/ciaran/mouse/mouse_plink --keep name_vector_train.txt --extract top_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out train_raw_plink_no_clump_"$1"_in_"$2"_"$3"

        plink1.9 --1 --prune --allow-no-sex --pheno $phenofile --bfile /external_storage/ciaran/mouse/mouse_plink --keep name_vector_test.txt --extract top_"$5"_snps_"$1"_in_"$2"_"$3".txt --recode A --out test_raw_plink_no_clump_"$1"_in_"$2"_"$3"


fi

