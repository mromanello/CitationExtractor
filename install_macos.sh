#!/bin/bash

# (this script should be ran with sudo -H)
if [ "$(id -u)" != "0" ]; then
   echo "This script must be run as root, try with sudo -H" 1>&2
   exit 1
fi

# Set the paths for the project and its dependencies
project_dir=$(pwd)
dependencies_dir=${project_dir}/..
echo "> CitationExtractor will be installed in: $project_dir"
echo "> CitationExtractor dependencies will be installed in: $dependencies_dir"

# Probably not necessary (?)
export C_INCLUDE_PATH=/usr/local/include/:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/include/:${CPLUS_INCLUDE_PATH}

# INSTALL CRF++
echo "> Installing crfpp"
cd $dependencies_dir
git clone https://github.com/taku910/crfpp.git
cd crfpp/
./configure
sed -i'' -e '/#include "winmain.h"/d' crf_test.cpp
sed -i'' -e '/#include "winmain.h"/d' crf_learn.cpp
make install
make clean
chmod -R 777 ../crfpp
pip install -e python

# INSTALL TREETAGGER
echo "> Installing tree-tagger"
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
echo "> Installing hucit_kb"
cd $dependencies_dir
git clone https://github.com/mromanello/hucit_kb.git
cd hucit_kb
pip install -U -r requirements.txt
pip install .

# INSTALL CITATION_PARSER
echo "> Installing CitationParser"
cd $dependencies_dir
git clone https://github.com/mromanello/CitationParser.git
cd CitationParser
python setup.py install

# INSTALL PROJECT (CITATION_EXTRACTOR)
echo "> Installing CitationExtractor"
cd $project_dir
pip install -e lib
pip install -U -r requirements.txt
pip install .
