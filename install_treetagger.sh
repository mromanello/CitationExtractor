#!/bin/bash

cd
mkdir tree-tagger
cd tree-tagger/
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.1.tar.gz
#wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-MacOSX-3.2.tar.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh
# parameter files for tagger
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/dutch.par.gz
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english.par.gz
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/french.par.gz
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/german.par.gz
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/italian.par.gz
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/latin.par.gz
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/portuguese.par.gz
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/spanish.par.gz
# parameter files for chunker
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english-chunker.par.gz
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/french-chunker.par.gz
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/german-chunker.par.gz
wget https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/spanish-chunker.par.gz
# let's run the installation script
chmod +x install-tagger.sh
chown $USER:$USER -R .
chmod 777 -R .
./install-tagger.sh
rm *.gz
