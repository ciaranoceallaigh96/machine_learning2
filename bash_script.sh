awk '{print $1, $1}' $1 > "$1"_double.txt
plink2 --assoc --bfile --extract --out 
#clump
plink2 --clump --params --out
head -n 10000 X.prune.in | awk '{print $X}' > top10ksnps.txt
#extract top snps
plink1.9 --bfile --keep --extract --recode A --out 


