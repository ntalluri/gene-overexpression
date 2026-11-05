These files were create using DataSAIL and mmseq2. 
- script/run_datasail_split.sh
- script/run_datasail_split.sub

run	train	val	test	total deviation	after-move counts
C1e_6	70.86	13.29	15.85	3.42	2901 / 544 / 649
C1e_12	71.81	14.75	13.43	3.63	2940 / 604 / 550
C1e_10	72.03	13.68	14.29	4.06	2949 / 560 / 585

I picked c1e_10 as this kept things very similar and even to a 70/15/15 split

train	val	test	total
as DataSAIL produced	2862 (69.91%)	604 (14.75%)	628 (15.34%)	4094
after moving families	2949 (72.03%)	560 (13.68%)	585 (14.29%)	4094

87 genes moved, 44 out of val and 43 out of test.

Leakage for this run specifically, from leakage_threshold.tsv:

	before	after
train vs val	2.98%	0.00%
train vs test	3.50%	0.00%
val vs test	1.43%	0.00%