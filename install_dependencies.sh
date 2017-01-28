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