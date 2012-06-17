# Settings for the citation extractor

# Sets debug on (=true) or off (=false)
DEBUG = False

# leave empty to write the log to the console
LOG_FILE = ""

# list of directories containing data (IOB format with .iob extension)
DATA_DIRS = (
	"data/",
	"data_C1/",
)

DATA_FILE = "aph_data_C1/all_in_one.iob"

TEMP_DIR = "tmp/"

OUTPUT_DIR = "out/"

# number of iterations for the k-fold cross validation
CROSS_VAL_FOLDS = 10

CRFPP_TEMPLATE_DIR = "crfpp_templates/"

CRFPP_TEMPLATE = "template_4.tpl"

# Leave empty to use CRF++'s default value
CRFPP_PARAM_C = ''

# Leave empty to use CRF++'s default value
CRFPP_PARAM_A = ''

# Leave empty to use CRF++'s default value
CRFPP_PARAM_F = ''