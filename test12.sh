# since batch-100's result are just the batch-1's result times 100, and batch-100's nvprof take a lot of time, so this .sh script just run batch-1 test.
# if you want to check other result, just change the '$bacth' variate blow to other value(alert again: nvprof will take tons of time for large batch)

batch=1
METRICS=gld_transactions,gst_transactions
# compare the main kernel

# small matrices
# ours svd main kernel
OUR_K0="--kernels small_svd_even_column --metrics $METRICS"
# cusolver main kernel
CU_K0="--kernels batched_svd_parallel_jacobi_32x16 --metrics $METRICS"

# ==========================================================================================================8
sudo nvprof --log-file temp1.txt $OUR_K0 ./test 11 $batch 8

str1=$(sed -n "9,9p" temp1.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp1.txt)
str2=${str2##* }
gt_ours=$(($str1+$str2))
echo "8x8, our svd, global memory transctions: $gt_ours"

sudo nvprof --log-file temp2.txt $CU_K0 ./test 12 $batch 8

str1=$(sed -n "9,9p" temp2.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp2.txt)
str2=${str2##* }
gt_cu=$(($str1+$str2))
echo "8x8, cusolver svd, global memory transctions: $gt_cu"
awk 'BEGIN{printf "ratio: %.2f\n",'$gt_ours'/'$gt_cu'}'

# ==========================================================================================================16
sudo nvprof --log-file temp1.txt $OUR_K0 ./test 11 $batch 16

str1=$(sed -n "9,9p" temp1.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp1.txt)
str2=${str2##* }
gt_ours=$(($str1+$str2))
echo "16x16, our svd, global memory transctions: $gt_ours"

sudo nvprof --log-file temp2.txt $CU_K0 ./test 12 $batch 16

str1=$(sed -n "9,9p" temp2.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp2.txt)
str2=${str2##* }
gt_cu=$(($str1+$str2))
echo "16x16, cusolver svd, global memory transctions: $gt_cu"
awk 'BEGIN{printf "ratio: %.2f\n",'$gt_ours'/'$gt_cu'}'

# ==========================================================================================================32
sudo nvprof --log-file temp1.txt $OUR_K0 ./test 11 $batch 32

str1=$(sed -n "9,9p" temp1.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp1.txt)
str2=${str2##* }
gt_ours=$(($str1+$str2))
echo "32x32, our svd, global memory transctions: $gt_ours"

sudo nvprof --log-file temp2.txt $CU_K0 ./test 12 $batch 32

str1=$(sed -n "9,9p" temp2.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp2.txt)
str2=${str2##* }
gt_cu=$(($str1+$str2))
echo "32x32, cusolver svd, global memory transctions: $gt_cu"
awk 'BEGIN{printf "ratio: %.2f\n",'$gt_ours'/'$gt_cu'}'

# large matrices
# our svd nvprof main kernel
OUR_K1="--kernels generate_jointG00 --metrics $METRICS"
OUR_K2="--kernels myevd_batched_16 --metrics $METRICS"
OUR_K3="--kernels updateBlockColumn2_16 --metrics $METRICS"

# cusolver nvprof main kernel
CU_K1="--kernels gesvdbj_batch_32x16 --metrics $METRICS"
CU_K2="--kernels svd_column_rotate_batch_32x16 --metrics $METRICS"
CU_K3="--kernels svd_row_rotate_batch_32x16 --metrics $METRICS"

# ==========================================================================================================64
sudo nvprof --log-file temp1.txt $OUR_K1 $OUR_K2 $OUR_K3 ./test 11 $batch 64

str1=$(sed -n "9,9p" temp1.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "12,12p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "13,13p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "15,15p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp1.txt)
str2=${str2#* }
str2=${str2%%g*}
gt_ours=$(($str1*$str2))
echo "64x64, our svd, global memory transctions: $gt_ours"

sudo nvprof --log-file temp2.txt $CU_K1 $CU_K2 $CU_K3 ./test 12 $batch 64

str1=$(sed -n "9,9p" temp2.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "12,12p" temp2.txt)
str2=${str2##* }
str2=$((2*$str2))
str1=$(($str1+$str2))
str2=$(sed -n "13,13p" temp2.txt)
str2=${str2##* }
str2=$((2*$str2))
str1=$(($str1+$str2))
str2=$(sed -n "15,15p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp2.txt)
str2=${str2#* }
str2=${str2%%g*}
gt_cu=$(($str1*$str2))
echo "64x64, cusolver svd, global memory transctions: $gt_cu"
awk 'BEGIN{printf "ratio: %.2f\n",'$gt_ours'/'$gt_cu'}'

# ==========================================================================================================128
sudo nvprof --log-file temp1.txt $OUR_K1 $OUR_K2 $OUR_K3 ./test 11 $batch 128

str1=$(sed -n "9,9p" temp1.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "12,12p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "13,13p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "15,15p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp1.txt)
str2=${str2#* }
str2=${str2%%g*}
gt_ours=$(($str1*$str2))
echo "128x128, our svd, global memory transctions: $gt_ours"

sudo nvprof --log-file temp2.txt $CU_K1 $CU_K2 $CU_K3 ./test 12 $batch 128

str1=$(sed -n "9,9p" temp2.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "12,12p" temp2.txt)
str2=${str2##* }
str2=$((2*$str2))
str1=$(($str1+$str2))
str2=$(sed -n "13,13p" temp2.txt)
str2=${str2##* }
str2=$((2*$str2))
str1=$(($str1+$str2))
str2=$(sed -n "15,15p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp2.txt)
str2=${str2#* }
str2=${str2%%g*}
gt_cu=$(($str1*$str2))
echo "128x128, cusolver svd, global memory transctions: $gt_cu"
awk 'BEGIN{printf "ratio: %.2f\n",'$gt_ours'/'$gt_cu'}'

# ==========================================================================================================256
sudo nvprof --log-file temp1.txt $OUR_K1 $OUR_K2 $OUR_K3 ./test 11 $batch 256

str1=$(sed -n "9,9p" temp1.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "12,12p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "13,13p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "15,15p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp1.txt)
str2=${str2#* }
str2=${str2%%g*}
gt_ours=$(($str1*$str2))
echo "256x256, our svd, global memory transctions: $gt_ours"

sudo nvprof --log-file temp2.txt $CU_K1 $CU_K2 $CU_K3 ./test 12 $batch 256

str1=$(sed -n "9,9p" temp2.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "12,12p" temp2.txt)
str2=${str2##* }
str2=$((2*$str2))
str1=$(($str1+$str2))
str2=$(sed -n "13,13p" temp2.txt)
str2=${str2##* }
str2=$((2*$str2))
str1=$(($str1+$str2))
str2=$(sed -n "15,15p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp2.txt)
str2=${str2#* }
str2=${str2%%g*}
gt_cu=$(($str1*$str2))
echo "256x256, cusolver svd, global memory transctions: $gt_cu"
awk 'BEGIN{printf "ratio: %.2f\n",'$gt_ours'/'$gt_cu'}'

# ==========================================================================================================512
sudo nvprof --log-file temp1.txt $OUR_K1 $OUR_K2 $OUR_K3 ./test 11 $batch 512

str1=$(sed -n "9,9p" temp1.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "12,12p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "13,13p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "15,15p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp1.txt)
str2=${str2#* }
str2=${str2%%g*}
gt_ours=$(($str1*$str2))
echo "512x512, our svd, global memory transctions: $gt_ours"

sudo nvprof --log-file temp2.txt $CU_K1 $CU_K2 $CU_K3 ./test 12 $batch 512

str1=$(sed -n "9,9p" temp2.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "12,12p" temp2.txt)
str2=${str2##* }
str2=$((2*$str2))
str1=$(($str1+$str2))
str2=$(sed -n "13,13p" temp2.txt)
str2=${str2##* }
str2=$((2*$str2))
str1=$(($str1+$str2))
str2=$(sed -n "15,15p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp2.txt)
str2=${str2#* }
str2=${str2%%g*}
gt_cu=$(($str1*$str2))
echo "512x512, cusolver svd, global memory transctions: $gt_cu"
awk 'BEGIN{printf "ratio: %.2f\n",'$gt_ours'/'$gt_cu'}'


# ==========================================================================================================1024

sudo nvprof --log-file temp1.txt $OUR_K1 $OUR_K2 $OUR_K3 ./test 11 $batch 1024

str1=$(sed -n "9,9p" temp1.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "12,12p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "13,13p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "15,15p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp1.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp1.txt)
str2=${str2#* }
str2=${str2%%g*}
gt_ours=$(($str1*$str2))
echo "1024x1024, our svd, global memory transctions: $gt_ours"

sudo nvprof --log-file temp2.txt $CU_K1 $CU_K2 $CU_K3 ./test 12 $batch 1024

str1=$(sed -n "9,9p" temp2.txt)
str1=${str1##* }
str2=$(sed -n "10,10p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "12,12p" temp2.txt)
str2=${str2##* }
str2=$((2*$str2))
str1=$(($str1+$str2))
str2=$(sed -n "13,13p" temp2.txt)
str2=${str2##* }
str2=$((2*$str2))
str1=$(($str1+$str2))
str2=$(sed -n "15,15p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp2.txt)
str2=${str2##* }
str1=$(($str1+$str2))
str2=$(sed -n "16,16p" temp2.txt)
str2=${str2#* }
str2=${str2%%g*}
gt_cu=$(($str1*$str2))
echo "1024x1024, cusolver svd, global memory transctions: $gt_cu"
awk 'BEGIN{printf "ratio: %.2f\n",'$gt_ours'/'$gt_cu'}'