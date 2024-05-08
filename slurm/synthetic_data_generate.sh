N=2

for i in $(seq 1 $N)
do
    sbatch synthetic_batch.sh "$i"
done