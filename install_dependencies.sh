#!/bin/bash
# (this script should be ran as root/with sudo!)
#
# INSTALL CRF++
git clone https://github.com/taku910/crfpp.git
cd crfpp/
./configure
sed -i '/#include "winmain.h"/d' crf_test.cpp
sed -i '/#include "winmain.h"/d' crf_learn.cpp
make install
make clean
ldconfig
#rm -fr ../crfpp
cd
mkdir tree-tagger
cd tree-tagger/
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.tar.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh
# parameter files for tagger
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/dutch-par-linux-3.2-utf8.bin.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english-par-linux-3.2-utf8.bin.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/french-par-linux-3.2-utf8.bin.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/german-par-linux-3.2-utf8.bin.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/italian-par-linux-3.2-utf8.bin.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/latin-par-linux-3.2.bin.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/portuguese-par-linux-3.2-utf8.bin.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/spanish-par-linux-3.2-utf8.bin.gz
# parameter files for chunker
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english-chunker-par-linux-3.2-utf8.bin.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/french-chunker-par-linux-3.2-utf8.bin.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/german-chunker-par-linux-3.2-utf8.bin.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/spanish-chunker-par-linux-3.2-utf8.bin.gz
# let's run the installation script
chmod +x install-tagger.sh 
./install-tagger.sh
rm *.gz
