## TODO: non-dependent tasks

- start writing feature functions (bigram, trirgam)
- write small test as a sanity check for each

- build viterbi
- evaluate

## TODO: 6.12

- validate loss on simple example OK
- complete gradient OK
- sanity check OK
- converge check OK
- complete predict prob (one of the inputs for viterbi) TODO
- sanity check TODO
- train on train.wtag TRIED(with unigram, and stopped, not efficient yet, see below)
- evaluate on individual tags (not all sentence) SKIPPED(done "analysis as sanity check" for the meantime)
- improve vectorized ops with sparse matrices TODO


## tests that should pass

# FIRST LOSS
(on train_dev.wtag, should produce feature matrix of size (29,45))

```python
0-29*np.log(45) == -110.39321220333923
```

# FIRST GRAD
```python
first_grad = [-0.64444444,  1.35555556,  3.35555556, - 0.64444444, - 0.64444444,  1.35555556,
             2.35555556, - 0.64444444,  0.35555556, - 0.64444444, - 0.64444444,  3.35555556,
             1.35555556,  1.35555556, - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444,
             - 0.64444444, - 0.64444444,  0.35555556, - 0.64444444, - 0.64444444, - 0.64444444,
             0.35555556, - 0.64444444, - 0.64444444,  1.35555556, - 0.64444444,  0.35555556,
             - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444,
             - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444, - 0.64444444,
             1.35555556,  1.35555556, - 0.64444444]
```


# warnings to take care
```python
# current warnings
postagger
#   ret = np.log(np.sum(np.exp(v.dot(y_matrix.T)))
postagger
#   numerator = np.exp(v.dot(y_matrix.T))
postagger
#   ret = numerator / denom
# https://stackoverflow.com/questions/4359959/overflow-in-exp-in-scipy-numpy-in-python
# https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
```


# ran on train.wtag 

- ran more than 1 hour and didn't converge, stopped and got to 29 iterations, output:

```python
[-5.00536623 11.15340433 11.84649125 -5.00536623 -5.00536623 11.15340433
 11.55882306 -5.00536623 10.46017933 -5.00536623 -5.00536623 11.84649125
 11.15340433 11.15340433 -5.00536623 -5.00536623 -5.00536623 -5.00536623
 -5.00536623 -5.00536623 10.46017933 -5.00536623 -5.00536623 -5.00536623
 10.46017933 -5.00536623 -5.00536623 11.15340433 -5.00536623 10.46017933
 -5.00536623 -5.00536623 -5.00536623 -5.00536623 -5.00536623 -5.00536623
 -5.00536623 -5.00536623 -5.00536623 -5.00536623 -5.00536623 -5.00536623
 11.15340433 11.15340433 -5.00536623]
```

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

# analysis as a sanity check

take a look at our tags:
```python
poss = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
        'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
        'WDT', 'WP', 'WP$', 'WRB', '#', '$', '\'\'', '``', '(', ')', ',', '.', ':']
 ```
 
 the fit output (our vector) is currently composed of unigram features and 1 iteration only:

observe our vector on different set

# train_dev
```python
[-0.4372312   0.91969321  2.27661763 -0.4372312  -0.4372312   0.91969321
  1.59815542 -0.4372312   0.24123101 -0.4372312  -0.4372312   2.27661763
  0.91969321  0.91969321 -0.4372312  -0.4372312  -0.4372312  -0.4372312
 -0.4372312  -0.4372312   0.24123101 -0.4372312  -0.4372312  -0.4372312
  0.24123101 -0.4372312  -0.4372312   0.91969321 -0.4372312   0.24123101
 -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312
 -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312  -0.4372312
  0.91969321  0.91969321 -0.4372312 ]
```

# train_dev_50
```python
[ 0.00613064  0.10162715  0.33506305 -0.1035135  -0.1035135   0.33152615
  0.16529149 -0.09290277 -0.08229205 -0.1035135  -0.06460751  0.5861835
  0.22541892  0.32091542 -0.1035135  -0.1035135  -0.07168133 -0.05045988
 -0.06460751  0.02027827 -0.08582896 -0.1035135  -0.09290277 -0.1035135
  0.02381518 -0.1035135   0.03088899  0.06272116  0.04503662 -0.00094318
 -0.01509081 -0.0115539  -0.0575337  -0.09290277 -0.09290277 -0.09643968
 -0.1035135  -0.0575337  -0.08229205 -0.08229205 -0.1035135  -0.1035135
  0.09455333  0.06979498 -0.09290277]
```

# train_dev_500
 ```python
[-0.00150417  0.06914268  0.30845427 -0.09767266 -0.10248109  0.36504573
  0.1856545  -0.09582327 -0.09397388 -0.10248109 -0.05069806  0.5348201
  0.16346178  0.41239022 -0.10248109 -0.10211121 -0.05439684 -0.02850533
 -0.07104139  0.05249813 -0.09471363 -0.09989194 -0.08509679 -0.10248109
  0.01440062 -0.10211121  0.03585359  0.0565668  -0.01666921  0.01403074
 -0.05106793  0.00367413 -0.0795486  -0.09138472 -0.1006317  -0.09286424
 -0.10248109 -0.069192   -0.07547994 -0.07474018 -0.10248109 -0.10248109
  0.12684377  0.07875953 -0.08768594]
```       

```
we can see how 'DT' gets more importance because it is more common in the first set of 2 sentences, and as we increase
the number of sentences its importances decreases.
