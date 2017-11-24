"""Settings for a MaxEnt-based citation extractor."""

import pkg_resources
from sklearn.linear_model import LogisticRegression

# Sets debug on (=true) or off (=false)
DEBUG = False
POS = True
# leave empty to write the log to the console
LOG_FILE = ""

# list of directories containing data (IOB format with .iob extension)
DATA_DIRS = (
    pkg_resources.resource_filename(
                'citation_extractor',
                'data/aph_corpus/goldset/iob/'
    ),
)

CLASSIFIER = CLASSIFIER = LogisticRegression()

TEST_DIR = ()

TRAIN_COLLECTIONS = ()

TEST_COLLECTIONS = ()

DATA_FILE = ""

TEMP_DIR = ""

OUTPUT_DIR = ""

# number of iterations for the k-fold cross validation
CROSS_VAL_FOLDS = 10

CRFPP_TEMPLATE_DIR = pkg_resources.resource_filename(
            'citation_extractor',
            'crfpp_templates/'
)

CRFPP_TEMPLATE = "template_5.tpl"
