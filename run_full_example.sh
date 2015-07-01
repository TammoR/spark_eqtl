/apps64/spark/sbin/stop-master.sh &
sleep 5

SPARK_LOG_DIR=~ /apps64/spark/sbin/start-master.sh & 
sleep 5

for i in {1..512}
do
   qsub -v SPARK_MASTER=spark://rbalhpc05:7077 -v xyz=spark://rbalhpc05:7077 ~/Dropbox/FastSpark.pbs
done


for i in {1..22} X
do
    python trans_analysis.py "full_chr$i" "chr$i"
    wait
    echo "Done with Chromosome AFR $i."
done
