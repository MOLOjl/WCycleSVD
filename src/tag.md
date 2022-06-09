test_tag array:

0:test tag 

1:result, our svd 

2:result, cusolver 

3:saved 
 

---
test tags:

|tag|means|
|:---:|:---|
|0|no stragety(tile_h=height)|
|1|experiential stragety|
|2|auto-tuning|
|3|no transform svd A_ij(in sm)|
|3.2|no transform svd A_ij(in gm)|
|4|specified tile_w|
|5|sepcified tile size|
|6|no transform evd(1-sweep) B_ij |
|7|one-round svd|
|8|one-round evd(full-sweep or 1-sweep)|
|9|specify accuracy|
|10|specify max-sweeps|
|11|return sweeps and rotation counts (no need time)|
|12|small matrix, 1 warp kernel|
