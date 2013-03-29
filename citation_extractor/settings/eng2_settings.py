# Settings for the citation extractor

# Sets debug on (=true) or off (=false)
DEBUG = False
POS = True
# leave empty to write the log to the console
LOG_FILE = ""

# list of directories containing data (IOB format with .iob extension)
DATA_DIRS = (
)

TEST_DIR = (
)

TRAIN_COLLECTIONS = (
	)

TEST_COLLECTIONS = (
)

DATA_FILE = ""

TEMP_DIR = "output/tmp/"

OUTPUT_DIR = "output/10fold/"

# number of iterations for the k-fold cross validation
CROSS_VAL_FOLDS = 10

CRFPP_TEMPLATE_DIR = "source/crex-1.2.4/citation_extractor/crfpp_templates/"

CRFPP_TEMPLATE = "template_6.tpl"

# Leave empty to use CRF++'s default value
CRFPP_PARAM_C = ''

# Leave empty to use CRF++'s default value
CRFPP_PARAM_A = ''

# Leave empty to use CRF++'s default value
CRFPP_PARAM_F = ''