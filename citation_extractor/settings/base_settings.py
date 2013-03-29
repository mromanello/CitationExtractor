# Settings for the citation extractor

# Sets debug on (=true) or off (=false)
#DEBUG = True
DEBUG = False

# leave empty to write the log to the console
LOG_FILE = ""

DATA_BASEDIR = "/Users/rromanello/Documents/APh_Corpus/by_collection/"

# list of directories containing data (IOB format with .iob extension)
dirs = ('C1'
			,'C2'
			,'eff_cand_1_a'
			,'eff_cand_1_b'
			,'eff_cand_2_a'
			,'eff_cand_2_b'
			,'eff_cand_3_a'
			,'eff_cand_3_b'
			,'eff_cand_4_a'
			,'eff_cand_4_b'
			,'eff_cand_5_a'
			,'eff_cand_5_b'
			,'eff_cand_6_a'
			,'eff_cand_6_b'
			,'eff_cand_7'
			,'eff_cand_8'
			,'eff_cand_9')
			
test_dirs = ('eff_cand_10',)
			
DATA_DIRS = ["%s%s/"%(DATA_BASEDIR,collection) for collection in dirs]

TEST_DIRS = ["%s%s/"%(DATA_BASEDIR,collection) for collection in test_dirs]

DATA_FILE = ""

OUTPUT_DIR = "/Users/rromanello/Documents/crex/citation_extractor/citation_extractor/output/"

TEMP_DIR = "/Users/rromanello/Documents/crex/citation_extractor/citation_extractor/output/"

# number of iterations for the k-fold cross validation
CROSS_VAL_FOLDS = 10

CRFPP_TEMPLATE_DIR = "/Users/rromanello/Documents/crex/citation_extractor/citation_extractor/crfpp_templates/"

CRFPP_TEMPLATE = "template_5.tpl"

# Leave empty to use CRF++'s default value
CRFPP_PARAM_C = ''

# Leave empty to use CRF++'s default value
CRFPP_PARAM_A = ''

# Leave empty to use CRF++'s default value
CRFPP_PARAM_F = ''