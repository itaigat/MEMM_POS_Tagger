## unigram, 1 iteration

# train_dev 500

```
Parsing iterables: 0.022192 s
Building feature matrix: 9.082817 s
'__init__'  9106.08 ms
Loss first term: 0.001300 s
Loss second term: 6.462507 s
Loss reg: 0.000011 s
Loss final sum: 0.000001 s
'loss'  6463.87 ms
Grad first term: 0.000553 s
Grad second term: 6.775561 s
Grad last sum: 0.000005 s
'grad'  6776.17 ms
Loss first term: 0.001982 s
Loss second term: 6.453682 s
Loss reg: 0.000012 s
Loss final sum: 0.000001 s
'loss'  6455.72 ms
Grad first term: 0.000525 s
Grad second term: 6.631229 s
Grad last sum: 0.000005 s
'grad'  6631.81 ms
[-0.00150417  0.06914268  0.30845427 -0.09767266 -0.10248109  0.36504573
  0.1856545  -0.09582327 -0.09397388 -0.10248109 -0.05069806  0.5348201
  0.16346178  0.41239022 -0.10248109 -0.10211121 -0.05439684 -0.02850533
 -0.07104139  0.05249813 -0.09471363 -0.09989194 -0.08509679 -0.10248109
  0.01440062 -0.10211121  0.03585359  0.0565668  -0.01666921  0.01403074
 -0.05106793  0.00367413 -0.0795486  -0.09138472 -0.1006317  -0.09286424
 -0.10248109 -0.069192   -0.07547994 -0.07474018 -0.10248109 -0.10248109
  0.12684377  0.07875953 -0.08768594]
(45,)
'fit'  26328.40 ms
```

# train_dev 50

```
Parsing iterables: 0.002398 s
Building feature matrix: 0.066304 s
'__init__'  68.90 ms
Loss first term: 0.000293 s
Loss second term: 0.766870 s
Loss reg: 0.000012 s
Loss final sum: 0.000001 s
'loss'  767.22 ms
Grad first term: 0.000080 s
Grad second term: 0.693533 s
Grad last sum: 0.000005 s
'grad'  693.67 ms
Loss first term: 0.000233 s
Loss second term: 0.726740 s
Loss reg: 0.000011 s
Loss final sum: 0.000001 s
'loss'  727.03 ms
Grad first term: 0.000066 s
Grad second term: 0.690917 s
Grad last sum: 0.000005 s
'grad'  691.03 ms
[ 0.00613064  0.10162715  0.33506305 -0.1035135  -0.1035135   0.33152615
  0.16529149 -0.09290277 -0.08229205 -0.1035135  -0.06460751  0.5861835
  0.22541892  0.32091542 -0.1035135  -0.1035135  -0.07168133 -0.05045988
 -0.06460751  0.02027827 -0.08582896 -0.1035135  -0.09290277 -0.1035135
  0.02381518 -0.1035135   0.03088899  0.06272116  0.04503662 -0.00094318
 -0.01509081 -0.0115539  -0.0575337  -0.09290277 -0.09290277 -0.09643968
 -0.1035135  -0.0575337  -0.08229205 -0.08229205 -0.1035135  -0.1035135
  0.09455333  0.06979498 -0.09290277]
(45,)
'fit'  2879.84 ms

Process finished with exit code 0
```

# train_dev

```
Parsing iterables: 0.000139 s
Building feature matrix: 0.000596 s
'__init__'  1.16 ms
Loss first term: 0.000112 s
Loss second term: 0.025774 s
Loss reg: 0.000014 s
Loss final sum: 0.000001 s
'loss'  25.94 ms
Grad first term: 0.000022 s
Grad second term: 0.033934 s
Grad last sum: 0.000008 s
'grad'  34.02 ms
Loss first term: 0.000024 s
Loss second term: 0.030354 s
Loss reg: 0.000015 s
Loss final sum: 0.000001 s
'loss'  30.44 ms
Grad first term: 0.000015 s
Grad second term: 0.018021 s
Grad last sum: 0.000005 s
'grad'  18.08 ms
Loss first term: 0.000013 s
Loss second term: 0.014998 s
Loss reg: 0.000011 s
Loss final sum: 0.000001 s
'loss'  15.05 ms
Grad first term: 0.000011 s
Grad second term: 0.015491 s
Grad last sum: 0.000005 s
'grad'  15.54 ms
[-0.4372312   0.91969321  2.27661763 -0.4372312  -0.4372312   0.91969321
  1.59815542 -0.4372312   0.24123101 -0.4372312  -0.4372312   2.27661763
  0.91969321  0.91969321 -0.4372312  -0.4372312  -0.4372312  -0.4372312
 -0.4372312  -0.4372312   0.24123101 -0.4372312  -0.4372312  -0.4372312
  0.24123101 -0.4372312  -0.4372312   0.91969321 -0.4372312   0.24123101
 -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312
 -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312
  0.91969321  0.91969321 -0.4372312 ]
(45,)
'fit'  140.03 ms
```
# Fully vectorized
# unigram, 1 iteration

# train_dev

```
Parsing iterables: 0.000135 s
Building feature matrix: 0.000581 s
Building y_x features matrix: 0.061893 s
'__init__'  62.65 ms
Loss first term: 0.000058 s
Loss second term: 0.000185 s
Loss reg: 0.000013 s
Loss final sum: 0.000001 s
'loss'  0.28 ms
Grad first term: 0.000018 s
Grad second term: 0.000213 s
Grad last sum: 0.000006 s
'grad'  0.27 ms
Loss first term: 0.000018 s
Loss second term: 0.000131 s
Loss reg: 0.000010 s
Loss final sum: 0.000001 s
'loss'  0.19 ms
Grad first term: 0.000011 s
Grad second term: 0.000203 s
Grad last sum: 0.000005 s
'grad'  0.24 ms
Loss first term: 0.000016 s
Loss second term: 0.000145 s
Loss reg: 0.000010 s
Loss final sum: 0.000001 s
'loss'  0.19 ms
Grad first term: 0.000011 s
Grad second term: 0.000200 s
Grad last sum: 0.000005 s
'grad'  0.24 ms
Loss first term: 0.000015 s
Loss second term: 0.000143 s
Loss reg: 0.000009 s
Loss final sum: 0.000001 s
'loss'  0.19 ms
Grad first term: 0.000010 s
Grad second term: 0.000198 s
Grad last sum: 0.000004 s
'grad'  0.24 ms
[-0.4372312   0.91969321  2.27661763 -0.4372312  -0.4372312   0.91969321
  1.59815542 -0.4372312   0.24123101 -0.4372312  -0.4372312   2.27661763
  0.91969321  0.91969321 -0.4372312  -0.4372312  -0.4372312  -0.4372312
 -0.4372312  -0.4372312   0.24123101 -0.4372312  -0.4372312  -0.4372312
  0.24123101 -0.4372312  -0.4372312   0.91969321 -0.4372312   0.24123101
 -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312
 -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312
  0.91969321  0.91969321 -0.4372312 ]
(45,)
'fit'  2.29 ms

Process finished with exit code 0

```

# train_dev_50

```
Parsing iterables: 0.002409 s
Building feature matrix: 0.072239 s
Building y_x features matrix: 237.764106 s
'__init__'  237838.83 ms
Loss first term: 0.000232 s
Loss second term: 0.013207 s
Loss reg: 0.000017 s
Loss final sum: 0.000002 s
'loss'  13.51 ms
Grad first term: 0.000091 s
Grad second term: 0.009650 s
Grad last sum: 0.000009 s
'grad'  9.80 ms
Loss first term: 0.000116 s
Loss second term: 0.007406 s
Loss reg: 0.000015 s
Loss final sum: 0.000002 s
'loss'  7.58 ms
Grad first term: 0.000078 s
Grad second term: 0.009726 s
Grad last sum: 0.000007 s
'grad'  9.85 ms
[ 0.00613064  0.10162715  0.33506305 -0.1035135  -0.1035135   0.33152615
  0.16529149 -0.09290277 -0.08229205 -0.1035135  -0.06460751  0.5861835
  0.22541892  0.32091542 -0.1035135  -0.1035135  -0.07168133 -0.05045988
 -0.06460751  0.02027827 -0.08582896 -0.1035135  -0.09290277 -0.1035135
  0.02381518 -0.1035135   0.03088899  0.06272116  0.04503662 -0.00094318
 -0.01509081 -0.0115539  -0.0575337  -0.09290277 -0.09290277 -0.09643968
 -0.1035135  -0.0575337  -0.08229205 -0.08229205 -0.1035135  -0.1035135
  0.09455333  0.06979498 -0.09290277]
(45,)
'fit'  41.90 ms
```

* notice 'init' takes the most time now (building y_x_matrix)