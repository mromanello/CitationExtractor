# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

from __future__ import print_function
import pdb
import logging
import os
import codecs
#import knowledge_base
import glob
import sys,pprint,re,string
import pandas as pd
from random import *
from pyCTS import CTS_URN
from .strmatching import StringUtils

#import citation_extractor
#import xml.dom.minidom as mdom

global logger
logger = logging.getLogger(__name__)

NIL_ENTITY = "urn:cts:GreekLatinLit:NIL"


def init_logger(log_file=None, loglevel=logging.DEBUG):
    """
    Initialise the logger
    """
    if(log_file !="" or log_file is not None):
        logging.basicConfig(
            filename=log_file
            ,level=loglevel,format='%(asctime)s - %(name)s - [%(levelname)s] %(message)s',filemode='w',datefmt='%a, %d %b %Y %H:%M:%S'
        )
        logger = logging.getLogger(__name__)
        logger.info("Logger initialised")
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(loglevel)
        ch = logging.StreamHandler()
        ch.setLevel(loglevel)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        logger.info("Logger initialised")
    return logger
