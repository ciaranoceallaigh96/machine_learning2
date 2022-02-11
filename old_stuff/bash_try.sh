echo "hellow"
modeltype=$(cat modeltype.txt)
if [ $modeltype == $1 ]; 
then
  exit
fi
echo "red"
rm modeltype.txt

echo "$1" > modeltype.txt
