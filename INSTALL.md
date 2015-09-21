# Installation notes

(This document assumes as working directory the root of the git repository)

## Local Dependencies

The following command installs a few dependencies that need to be installed from local files (these files are shipped together as part of the `citation_extractor`):

    pip install -e lib/

## CRF++

Instructions for installing CRF++ can also be found here: <https://taku910.github.io/crfpp/#install>.

### Install CRF++

Get the installation files (.tar.gz. archive):

    wget "https://googledrive.com/host/0B4y35FiV1wh7fngteFhHQUN2Y1B5eUJBNHZUemJYQV9VWlBUb3JlX0xBdWVZTWtSbVBneU0/CRF++-0.58.tar.gz" -O CRF++.tar.gz

Uncompress, compile and install:

    tar -xzf CRF++.tar.gz
    cd CRF++-0.58/
    ./configure
    make
    make install

### Install the CRF++ python bundle

From within the `CRF++-0.58` directory:

    pip install -e python/

## Download the training/evaluation data

TODO


