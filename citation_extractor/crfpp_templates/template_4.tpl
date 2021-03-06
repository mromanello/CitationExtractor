# model for canonical references training
# currently the best performing template
# * removed prefixes and suffixes
# * removed pattern features for n-2..n+2
# version 1.2.1

#unigram
U01:%x[-2,0]
U02:%x[-1,0]
U03:%x[0,0]
U04:%x[1,0]
U05:%x[2,0]

# punctuation
U06:%x[-2,1]
U07:%x[-1,1]
U08:%x[0,1]
U09:%x[1,1]
U10:%x[2,1]

# brackets
U11:%x[-2,2]
U12:%x[-1,2]
U13:%x[0,2]
U14:%x[1,2]
U15:%x[2,2]

# case
U16:%x[-2,3]
U17:%x[-1,3]
U18:%x[0,3]
U19:%x[1,3]
U20:%x[2,3]

# number feature
U21:%x[-2,4]
U22:%x[-1,4]
U23:%x[0,4]
U24:%x[1,4]
U25:%x[2,4]

# pattern
U27:%x[0,15]

#  compressed pattern
U32:%x[0,16]

#  string length
#U35:%x[0,14]

# 4 prefixes
#U36:%x[0,5]
#U37:%x[0,6]
#U38:%x[0,7]
#U39:%x[0,8]

# 4 suffixes
#U40:%x[0,9]
#U41:%x[0,10]
#U42:%x[0,11]
#U43:%x[0,12]

# lowercase no punct
U46:%x[0,13]

# unigrams within context
U49:%x[-2,0]/%x[-1,0]/%x[0,0]
U50:%x[-1,0]/%x[0,0]/%x[1,0]
U51:%x[0,0]/%x[1,0]/%x[2,0]

# number feature within context
U52:%x[-2,4]/%x[-1,4]/%x[0,4]
U53:%x[-1,4]/%x[0,4]/%x[1,4]
U54:%x[0,4]/%x[1,4]/%x[2,4]

#  authors/works dictionary
U55:%x[-2,17]
U56:%x[-1,17]
U57:%x[0,17]
U58:%x[1,17]
U59:%x[2,17]

# bigram template
B0