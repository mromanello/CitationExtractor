import csv
newdict = {}

# read the csv catalog as a dictionary
with open ('catalog.csv','rb') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
        newdict[row['id']]=row
print newdict

# abstract ids grouped by language
bylang = {}
for id in newdict.keys():
    if(bylang.has_key(newdict[id]['lang'])):
        bylang[newdict[id]['lang']]+=[id]
    else:
        bylang[newdict[id]['lang']]=[]
        bylang[newdict[id]['lang']]+=[id]
print bylang

bycoll = {}
for id in newdict.keys():
    if(bycoll.has_key(newdict[id]['collection'])):
        bycoll[newdict[id]['collection']]+=[id]
    else:
        bycoll[newdict[id]['collection']]=[]
        bycoll[newdict[id]['collection']]+=[id]
print bycoll

# prints language and number of abstracts for each language
total = 0
for lang in bylang.keys():
	print lang,len(bylang[lang])
	for id in bylang[lang]:
		print id
	total += len(bylang[lang])
print total

# check that all the files have a matching entry in the catalog
for id in newdict.keys():
    try:
        file = open("txt/%s.txt"%id,"r")
    except Exception, e:
        print id
    try:
        file = open("iob/%s.txt"%id,"r")
    except Exception, e:
        print id,newdict[id]['collection']


# the following lines copy 
import os
import shutil

dir = "by_collection"
os.mkdir(dir)
for coll in bycoll:
    os.mkdir("%s/%s"%(dir,coll))
for coll in bycoll:
    for file in bycoll[coll]:
        try:
            shutil.copy("iob/%s.txt"%file,"%s%s/%s.iob"%("by_collection/",coll,file))
        except Exception, e:
            print file,e