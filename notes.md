Optimization succeeded.
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
(45,)


# FIRST LOSS
(on train_dev.wtag, should produce feature matrix of size (29,45))

0-29*np.log(45) == -110.39321220333923

# FIRST GRAD
inside tests

## TODO: 6.12

- validate loss on simple example OK
- complete gradient OK
- sanity check OK
- converge check OK
- complete predict prob
- sanity check
- train on train.wtag
- evaluate on individual tags (not all sentence)

- start building features
- bigram, trigam
- all of the rest from paper


Other non-dependent tasks
- build viterbi
- evaluate




```python
# current warnings
# /home/deebee/PycharmProjects/MEMM_POS_Tagger/src/utils/classifier.py:99: RuntimeWarning: overflow encountered in exp
#   ret = np.log(np.sum(np.exp(v.dot(y_matrix.T)))
# /home/deebee/PycharmProjects/MEMM_POS_Tagger/src/utils/classifier.py:150: RuntimeWarning: overflow encountered in exp
#   numerator = np.exp(v.dot(y_matrix.T))
# /home/deebee/PycharmProjects/MEMM_POS_Tagger/src/utils/classifier.py:153: RuntimeWarning: invalid value encountered in true_divide
#   ret = numerator / denom
# https://stackoverflow.com/questions/4359959/overflow-in-exp-in-scipy-numpy-in-python
# https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
```


# ran on train.wtag 

- ran more than 1 hour and didn't converge, got to 29 iterations, output:

KeyboardInterrupt
RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =           45     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.63709D+05    |proj g|=  1.42320D+04

At iterate    1    f=  4.39068D+05    |proj g|=  1.23282D+04

At iterate    2    f=  4.01376D+05    |proj g|=  2.66428D+04

At iterate    3    f=  3.79896D+05    |proj g|=  8.64502D+03

At iterate    4    f=  3.73130D+05    |proj g|=  4.87103D+03

At iterate    5    f=  3.70526D+05    |proj g|=  6.96556D+02

At iterate    6    f=  3.68960D+05    |proj g|=  2.18981D+03

At iterate    7    f=  3.66790D+05    |proj g|=  9.46167D+02

At iterate    8    f=  3.65950D+05    |proj g|=  1.54644D+03

At iterate    9    f=  3.65404D+05    |proj g|=  8.37514D+02

At iterate   10    f=  3.65238D+05    |proj g|=  2.88019D+02

At iterate   11    f=  3.64983D+05    |proj g|=  5.69508D+02

At iterate   12    f=  3.64843D+05    |proj g|=  3.94589D+02

At iterate   13    f=  3.64786D+05    |proj g|=  1.09652D+03

At iterate   14    f=  3.64656D+05    |proj g|=  1.68874D+02

At iterate   15    f=  3.64616D+05    |proj g|=  1.46037D+02

At iterate   16    f=  3.64548D+05    |proj g|=  2.44213D+02

At iterate   17    f=  3.64475D+05    |proj g|=  6.79143D+02

At iterate   18    f=  3.64427D+05    |proj g|=  4.81807D+02

At iterate   19    f=  3.64397D+05    |proj g|=  1.08427D+02

At iterate   20    f=  3.64375D+05    |proj g|=  1.71670D+02

At iterate   21    f=  3.64352D+05    |proj g|=  2.01931D+02

At iterate   22    f=  3.64320D+05    |proj g|=  2.30386D+02

At iterate   23    f=  3.64311D+05    |proj g|=  1.47048D+02

At iterate   24    f=  3.64304D+05    |proj g|=  4.98363D+01

At iterate   25    f=  3.64298D+05    |proj g|=  5.83120D+01

At iterate   26    f=  3.64288D+05    |proj g|=  8.07373D+01

At iterate   27    f=  3.64281D+05    |proj g|=  3.38964D+02

At iterate   28    f=  3.64267D+05    |proj g|=  1.11467D+02
 This problem is unconstrained.


- we can see the loss still goes down

# limit for 1 iteration
# ran on train_dev_500, output:

```python
'__init__'  8832.56 ms
'loss'  6735.76 ms
'grad'  6700.84 ms
'loss'  6600.88 ms
'grad'  6687.69 ms
[-0.00150417  0.06914268  0.30845427 -0.09767266 -0.10248109  0.36504573
  0.1856545  -0.09582327 -0.09397388 -0.10248109 -0.05069806  0.5348201
  0.16346178  0.41239022 -0.10248109 -0.10211121 -0.05439684 -0.02850533
 -0.07104139  0.05249813 -0.09471363 -0.09989194 -0.08509679 -0.10248109
  0.01440062 -0.10211121  0.03585359  0.0565668  -0.01666921  0.01403074
 -0.05106793  0.00367413 -0.0795486  -0.09138472 -0.1006317  -0.09286424
 -0.10248109 -0.069192   -0.07547994 -0.07474018 -0.10248109 -0.10248109
  0.12684377  0.07875953 -0.08768594]
(45,)
```

RUNNING THE L-BFGS-B CODE

           * * *

Machine precision = 2.220D-16
 N =           45     M =           10

At X0         0 variables are exactly at the bounds

At iterate    0    f=  4.74615D+04    |proj g|=  1.44593D+03

At iterate    1    f=  4.49109D+04    |proj g|=  1.25565D+03

           * * *

Tit   = total number of iterations
Tnf   = total number of function evaluations
Tnint = total number of segments explored during Cauchy searches
Skip  = number of BFGS updates skipped
Nact  = number of active bounds at final generalized Cauchy point
Projg = norm of the final projected gradient
F     = final function value

           * * *

   N    Tit     Tnf  Tnint  Skip  Nact     Projg        F
   45      1      2      1     0     0   1.256D+03   4.491D+04
  F =   44910.915825609532     

STOP: TOTAL NO. of ITERATIONS REACHED LIMIT                 

 Cauchy                time 0.000E+00 seconds.
 Subspace minimization time 0.000E+00 seconds.
 Line search           time 0.000E+00 seconds.

 Total User time 0.000E+00 seconds.

 This problem is unconstrained.

Process finished with exit code 0
