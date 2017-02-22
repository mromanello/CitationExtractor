#!/bin/bash
# (this script should be ran with sudo -H)

project_dir=$(pwd)
dependencies_dir=${project_dir}/..

export C_INCLUDE_PATH=/usr/local/include/:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/include/:${CPLUS_INCLUDE_PATH}

# INSTALL CRF++
cd $dependencies_dir
git clone https://github.com/taku910/crfpp.git
cd crfpp/
./configure
sed -i'' -e '/#include "winmain.h"/d' crf_test.cpp
sed -i'' -e '/#include "winmain.h"/d' crf_learn.cpp
make install
make clean
#ldconfig
chmod -R 777 ../crfpp
pip install -e python

# INSTALL TREETAGGER
cd $dependencies_dir
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


# INSTALL HUCIT KB
cd $dependencies_dir
git clone https://github.com/mromanello/hucit_kb.git
cd hucit_kb
pip install -r requirements.txt
pip install .

# INSTALL CITATION_PARSER
cd $dependencies_dir
git clone https://github.com/mromanello/CitationParser.git
cd CitationParser
python setup.py install

# INSTALL PROJECT (CITATION_EXTRACTOR)
cd $project_dir
pip install -e lib
pip install -U -r requirements.txt
pip install .
