# batch=10
str_1="| 10   "
echo "running batch 10x64x64 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 10 64
str1=$(cat temp.txt|tail -n 1)
str1=${str1##* }

echo "running batch 10x128x128 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 10 128
str2=$(cat temp.txt|tail -n 1)
str2=${str2##* }

echo "running batch 10x256x256 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 10 256
str3=$(cat temp.txt|tail -n 1)
str3=${str3##* }

echo "running batch 10x512x512 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 10 512
str4=$(cat temp.txt|tail -n 1)
str4=${str4##* }

echo "running batch 10x1024x1024 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 10 1024
str5=$(cat temp.txt|tail -n 1)
str5=${str5##* }

str_2="| 100  "
echo "running batch 100x64x64 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 100 64
str6=$(cat temp.txt|tail -n 1)
str6=${str6##* }

echo "running batch 100x128x128 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 100 128
str7=$(cat temp.txt|tail -n 1)
str7=${str7##* }

echo "running batch 100x256x256 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 100 256
str8=$(cat temp.txt|tail -n 1)
str8=${str8##* }

echo "running batch 100x512x512 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 100 512
str9=$(cat temp.txt|tail -n 1)
str9=${str9##* }

echo "running batch 100x1024x1024 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 100 1024
str10=$(cat temp.txt|tail -n 1)
str10=${str10##* }

str_3="| 500  "
echo "running batch 500x64x64 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 500 64
str11=$(cat temp.txt|tail -n 1)
str11=${str11##* }

echo "running batch 500x128x128 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 500 128
str12=$(cat temp.txt|tail -n 1)
str12=${str12##* }

echo "running batch 500x256x256 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 500 256
str13=$(cat temp.txt|tail -n 1)
str13=${str13##* }

echo "running batch 500x512x512 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 500 512
str14=$(cat temp.txt|tail -n 1)
str14=${str14##* }

echo "running batch 500x1024x1024 case"
sudo nvprof --log-file temp.txt --kernels myevd_batched_16 --metrics achieved_occupancy ./test 11 500 1024
str15=$(cat temp.txt|tail -n 1)
str15=${str15##* }

# print result
echo "|---------------------------------------------------------|"
echo "|      |                   matrix size                    |"
echo "|batch |--------------------------------------------------|"
echo "|------|64x64    |128x128  |256x256  |512x512  |1024x1024 |"
echo "$str_1|$str1 |$str2 |$str3 |$str4 |$str5  |"
echo "$str_2|$str6 |$str7 |$str8 |$str9 |$str10  |"
echo "$str_3|$str11 |$str12 |$str13 |$str14 |$str15  |"
echo "|---------------------------------------------------------|"
