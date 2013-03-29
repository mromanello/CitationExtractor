# The CRF++ template for the baseline
# with only lexical features and POS tags
# version 1.2.4

#unigram
U01:%x[-2,0]
U02:%x[-1,0]
U03:%x[0,0]
U04:%x[1,0]
U05:%x[2,0]

# unigrams within context
U06:%x[-2,0]/%x[-1,0]/%x[0,0]
U07:%x[-1,0]/%x[0,0]/%x[1,0]
U08:%x[0,0]/%x[1,0]/%x[2,0]

#POS tag
U60:%x[0,18]
#U61:%x[-2,18]/%x[-1,18]/%x[0,18]
#U62:%x[-1,18]/%x[0,18]/%x[1,18]
#U63:%x[0,18]/%x[1,18]/%x[2,18]