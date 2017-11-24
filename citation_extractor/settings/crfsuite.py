# Settings for the citation extractor

from sklearn_crfsuite import CRF
import pkg_resources

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

CLASSIFIER = CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

TEST_DIR = ()

TRAIN_COLLECTIONS = ()

TEST_COLLECTIONS = ()

DATA_FILE = ""

TEMP_DIR = ""

OUTPUT_DIR = ""

# number of iterations for the k-fold cross validation
CROSS_VAL_FOLDS = 10

CRFPP_TEMPLATE_DIR = pkg_resources.resource_filename('citation_extractor','crfpp_templates/')

CRFPP_TEMPLATE = "template_5.tpl"

# Leave empty to use CRF++'s default value
CRFPP_PARAM_C = ''

# Leave empty to use CRF++'s default value
CRFPP_PARAM_A = ''

# Leave empty to use CRF++'s default value
CRFPP_PARAM_F = ''
