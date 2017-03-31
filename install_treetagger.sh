#!/bin/bash

cd
mkdir tree-tagger
cd tree-tagger/
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.1.tar.gz
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
#chown $USER:$USER -R .
#chmod 777 -R . 
./install-tagger.sh
rm *.gz
