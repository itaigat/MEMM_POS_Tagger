tags in train2
['WP','JJS','LS','RBS','SYM','RBR'
'NNP','JJR','RP','EX','RB','PRP'
'VBP','IN','NN','DT','JJ','MD'
'VB','TO','NNS','VBD','CC','WDT'
'VBG','VBN','VBZ','CD','FW','WRB','PDT'
',',':','.']

tags in train2 - train

LS, SYM, FW

tags in train - train2

'``', '-LRB-', 'WP$', 'POS', '-RRB-', '\\'\\'', 'PRP$', 'UH', '$'


tags in test

140531227005424 = {str} 'FW'
140531229383288 = {str} 'WP$'
140531229649248 = {str} '-LRB-'
140531229649696 = {str} '-RRB-'
140531229718712 = {str} 'NNPS'
140531229856968 = {str} 'PRP'
140531229858368 = {str} 'VBN'
140531229872568 = {str} 'WDT'
140531229872904 = {str} 'JJS'
140531229880368 = {str} '``'
140531229880648 = {str} 'MD'
140531229884120 = {str} 'JJR'
140531229888440 = {str} 'RP'
140531229893440 = {str} 'VBP'
140531229901352 = {str} 'WP'
140531229902192 = {str} 'VBZ'
140531229904208 = {str} 'RBR'
140531229934064 = {str} '\\'\\''
140531229934624 = {str} 'WRB'
140531229934736 = {str} 'PDT'
140531229957344 = {str} 'EX'
140531230029224 = {str} 'RBS'
140531230039488 = {str} 'UH'
140531243926784 = {str} 'IN'
140531243926896 = {str} 'CD'
140531243926952 = {str} 'NNS'
140531243927064 = {str} 'VBG'
140531243927176 = {str} 'TO'
140531243927288 = {str} 'VB'
140531243927400 = {str} 'PRP$'
140531243927512 = {str} 'NN'
140531243927736 = {str} 'DT'
140531243927848 = {str} 'NNP'
140531243928072 = {str} 'POS'
140531243928128 = {str} 'JJ'
140531243928408 = {str} 'VBD'
140531243928464 = {str} 'RB'
140531243929528 = {str} 'CC'
140532397749672 = {str} ','
140532397957168 = {str} '$'
140532398368504 = {str} ':'
140532398778384 = {str} '#'
140532398809192 = {str} '.'


Model (1)

, 
('reg', 5): {'test_acc': 0.9111683703641126, 'train_acc': 0.9322004679226696},
{('reg', 3): {'test_acc': 0.9167863478922024, 'train_acc': 0.94060665763658} 
('reg', 1): {'test_acc': 0.9228689701782546, 'train_acc': 0.9544144809752494}, 
('reg', 0.1): {'test_acc': 0.9251077131029821, 'train_acc': 0.965718507572959}, 
('reg', 0.01): {'test_acc': 0.9243051448846836, 'train_acc': 0.9685670894389032}, 
('reg', 0.05): {'test_acc': 0.9252766748331502, 'train_acc': 0.9668760004925502}, 
('reg', 0.005): {'test_acc': 0.9239249809918053, 'train_acc': 0.9685424619299757}}


Model (2)

{('reg', 0.01): {'train_acc': 0.9917198660008849, 'validation_acc': 0.9417773237997957}, 
('reg', 0.1): {'train_acc': 0.9884963023829088, 'validation_acc': 0.9361593462717058}, 
('reg', 1): {'train_acc': 0.9754756336514759, 'validation_acc': 0.9284984678243106}, 
('reg', 3): {'train_acc': 0.9551229378673914, 'validation_acc': 0.9193054136874361}, 
('reg', 5): {'train_acc': 0.9381834270905758, 'validation_acc': 0.9065372829417774}, 
('reg', 10): {'train_acc': 0.9113836040705392, 'validation_acc': 0.8871297242083759}, 
('reg', 25): {'train_acc': 0.8660008848998167, 'validation_acc': 0.8467824310520939}, 
('reg', 50): {'train_acc': 0.8197964730421592, 'validation_acc': 0.8084780388151175}, 


('reg', 100): {'train_acc': 0.7586751785601415, 'validation_acc': 0.7405515832482125}, 
('reg', 500): {'train_acc': 0.45818848366095694, 'validation_acc': 0.4458631256384065}, 
('reg', 1000): {'train_acc': 0.3453637570317932, 'validation_acc': 0.34167517875383047}}

with cut

Current results {
	('reg', 0.001, "{'previousword-f106': 1, 'nextword-f107': 1, 'pre_pre_word': 0, 'next_next_word': 0}"): {
		'train_acc': 0.9895076164591365,
		'validation_acc': 0.9366700715015321
	},
	('reg', 0.01, "{'previousword-f106': 1, 'nextword-f107': 1, 'pre_pre_word': 0, 'next_next_word': 0}"): {
		'train_acc': 0.988559509512673,
		'validation_acc': 0.9387129724208376
	},
	('reg', 0.1, "{'previousword-f106': 1, 'nextword-f107': 1, 'pre_pre_word': 0, 'next_next_word': 0}"): {
		'train_acc': 0.9852727387649327,
		'validation_acc': 0.9376915219611849
	},
	('reg', 0.001, "{'previousword-f106': 1, 'nextword-f107': 1, 'pre_pre_word': 1, 'next_next_word': 1}"): {   
		'train_acc': 0.9826180393148347,
		'validation_acc': 0.9356486210418795
	},
	('reg', 0.01, "{'previousword-f106': 1, 'nextword-f107': 1, 'pre_pre_word': 1, 'next_next_word': 1}"): {
		'train_acc': 0.982681246444599,
		'validation_acc': 0.933605720122574
	},
	('reg', 0.1, "{'previousword-f106': 1, 'nextword-f107': 1, 'pre_pre_word': 1, 'next_next_word': 1}"): {
		'train_acc': 0.9808482396814361,
		'validation_acc': 0.9351378958120531
	},
	('reg', 0.001, "{'previousword-f106': 1, 'nextword-f107': 1, 'pre_pre_word': 2, 'next_next_word': 2}"): {
		'train_acc': 0.9811642753302573,
		'validation_acc': 0.9361593462717058
	},
	('reg', 0.01, "{'previousword-f106': 1, 'nextword-f107': 1, 'pre_pre_word': 2, 'next_next_word': 2}"): {
		'train_acc': 0.9809746539409645,
		'validation_acc': 0.9356486210418795
	},
	('reg', 0.1, "{'previousword-f106': 1, 'nextword-f107': 1, 'pre_pre_word': 2, 'next_next_word': 2}"): {
		'train_acc': 0.9795840970861514,
		'validation_acc': 0.9382022471910112
	},
	('reg', 0.001, "{'previousword-f106': 0, 'nextword-f107': 0, 'pre_pre_word': 2, 'next_next_word': 2}"): {
		'train_acc': 0.9884330952531446,
		'validation_acc': 0.9397344228804902
	},
	('reg', 0.01, "{'previousword-f106': 0, 'nextword-f107': 0, 'pre_pre_word': 2, 'next_next_word': 2}"): {
		'train_acc': 0.987484988306681,
		'validation_acc': 0.9402451481103167
	},
	('reg', 0.1, "{'previousword-f106': 0, 'nextword-f107': 0, 'pre_pre_word': 2, 'next_next_word': 2}"): {
		'train_acc': 0.984830288856583,
		'validation_acc': 0.9376915219611849
	},
	('reg', 0.001, "{'previousword-f106': 0, 'nextword-f107': 0, 'pre_pre_word': 1, 'next_next_word': 1}"): {
		'train_acc': 0.9889387522912585,
		'validation_acc': 0.9397344228804902
	},
	('reg', 0.01, "{'previousword-f106': 0, 'nextword-f107': 0, 'pre_pre_word': 1, 'next_next_word': 1}"): {
		'train_acc': 0.9882434738638518,
		'validation_acc': 0.9402451481103167
	},
	('reg', 0.1, "{'previousword-f106': 0, 'nextword-f107': 0, 'pre_pre_word': 1, 'next_next_word': 1}"): {
		'train_acc': 0.98508311737564,
		'validation_acc': 0.9361593462717058
	}
}