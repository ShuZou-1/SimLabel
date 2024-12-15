for((i=1;i<=10;i++));  
do   
echo "-------------------------"
b=5
number_sim=`expr $i \* $b`
echo "$number_sim"
python3 build_simclass.py --number_sim ${number_sim}
python3 HO_simneglabel.py --number_sim ${number_sim}
echo "-------------------------"
done
