# model for canonical references training
#unigram
U01:%x[-2,0]
U02:%x[-1,0]
U03:%x[0,0]
U04:%x[1,0]
U05:%x[2,0]

# bracket feature
U06:%x[-2,1]
U07:%x[-1,1]
U08:%x[0,1]
U09:%x[1,1]
U10:%x[2,1]

# case feature
U11:%x[0,2]

lowercase unigram
#U12:%x[-2,3]
U13:%x[1,3]
U14:%x[0,3]
U15:%x[1,3]
#U16:%x[2,3]

# number feature
U17:%x[-2,4]
U18:%x[-1,4]
U19:%x[0,4]
U20:%x[1,4]
U21:%x[2,4]

# lowercase no punct string length
#U22:%x[0,6]
#U23:%x[-2,5]
#U24:%x[-1,5]
#U25:%x[0,5]
#U26:%x[1,5]
#U27:%x[2,5]

# first 4 chars
U28:%x[0,7]
U29:%x[0,8]
U30:%x[0,9]
U31:%x[0,10]

#last 4 chars
U32:%x[0,11]
U33:%x[0,12]
U34:%x[0,13]
U35:%x[0,14]

# unigrams within context
U36:%x[-2,0]/%x[-1,0]/%x[0,0]
U37:%x[-1,0]/%x[0,0]/%x[1,0]
U38:%x[0,0]/%x[1,0]/%x[2,0]

# number feature within context
U39:%x[-2,4]/%x[-1,4]/%x[0,4]
U40:%x[-1,4]/%x[0,4]/%x[1,4]
U41:%x[0,4]/%x[1,4]/%x[2,4]

# bigram template
B0