#!/bin/bash
# (this script should be ran as root/with sudo!)
#
# INSTALL CRF++
wget "https://googledrive.com/host/0B4y35FiV1wh7fngteFhHQUN2Y1B5eUJBNHZUemJYQV9VWlBUb3JlX0xBdWVZTWtSbVBneU0/CRF++-0.58.tar.gz" -O CRF++.tar.gz
tar -xzf CRF++.tar.gz
chmod 777 -R CRF++-0.58/ 
cd CRF++-0.58/
./configure
make
make install
make clean
ldconfig
rm ../CRF++.tar.gz