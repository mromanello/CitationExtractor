# -*- coding: utf-8 -*-
# author: Matteo Romanello, matteo.romanello@gmail.com

import pdb
import getopt
from ConfigParser import SafeConfigParser
import os,re,string,logging,pprint,types,xmlrpclib,json
import citation_extractor
from citation_extractor.crfpp_wrap import *
from citation_extractor.Utils.IO import *
from sklearn_crfsuite import CRF
from stop_words import safe_get_stop_words

"""
This module contains the core of the citation extractor.
"""
global logger
logger = logging.getLogger(__name__)

def determine_path():
    """
    Borrowed from wxglade.py
    TODO move to Utils module
    """
    try:
        root = __file__
        if os.path.islink (root):
            root = os.path.realpath (root)
        return os.path.dirname (os.path.abspath (root))
    except:
        print "I'm sorry, but something is wrong."
        print "There is no __file__ variable. Please contact the author."
        sys.exit ()

class CRFPP_Classifier:
    """
    This class should extend an abstract classifier
    """
    def __init__(self,train_file_name,template_file_name,dir):
        #dir=determine_path()+"/data/"
        fe = FeatureExtractor()
        path,fn = os.path.split(train_file_name)
        train_fname=dir+fn+'.train'
        t = fe.prepare_for_training(train_file_name)
        out=open(train_fname,'w').write(t.encode("utf-8"))
        # TODO the .mdl file should go somewhere else
        model_fname=dir+fn+'.mdl'
        template_fname = template_file_name
        train_crfpp(template_fname,train_fname,model_fname)
        self.crf_model=CRF_classifier(model_fname)
        return

    def classify(self,feature_sets):
        """
        Args:
            feature_sets:
                a list of dictionaries like the following:

                [{'a_token': u'Nella',
                 'b_punct': 'OTHERS',
                 'c_brackets': 'OTHERS',
                 'd_case': 'INIT_CAPS',
                 'e_number': 'NO_DIGITS',
                 'f_ngram_1': u'N',
                 'f_ngram_2': u'Ne',
                 'f_ngram_3': u'Nel',
                 'f_ngram_4': u'Nell',
                 'g_ngram_1': u'a',
                 'g_ngram_2': u'la',
                 'g_ngram_3': u'lla',
                 'g_ngram_4': u'ella',
                 'h_lowcase': u'nella',
                 'i_str-length': '5',
                 'l_pattern': 'Aaaaa',
                 'm_compressed-pattern': 'Aa',
                 'n_works_dictionary': 'OTHERS',
                 'z': '_'}, ... ]

        Returns:
            result:
                a list of dictionaries, where each dictionary corresponds
                to a token,

                [{'features': [],
                 'id': 37,
                 'label': 'O',
                 'probs': {'B-AAUTHOR':
                     {'alpha': '234.113833',
                     'beta': '-2.125040',
                     'prob': '0.000262'},
                   },
                 'token': '.'},...]
        """
        tagged_tokens_list = instance_to_string(feature_sets)
        return self.crf_model.classify(tagged_tokens_list)

class ScikitClassifierAdapter:
    """
    An adapter for an SklearnClassifier (nltk.classify.scikitlearn) object
    to make sure that all classifiers take same input and return the same output
    and are trained in the same way.

    scikit_classifier:
        a Scikit classifier *instance*

    train_file_name:
        the path to the training settings

    template_file_name:
        the template to extract additional feature for optimization purposes

    """
    def __init__(self, scikit_classifier, train_file_name,template_file_name,labelled_feature_sets=None):
        from nltk.classify.scikitlearn import SklearnClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import RandomForestClassifier
        fe = FeatureExtractor()
        logger.info("Initialising SklearnClassifier with model %s" %\
                scikit_classifier)
        #self.classifier = SklearnClassifier(scikit_classifier,sparse=False)
        if(isinstance(scikit_classifier,RandomForestClassifier)):
            self.classifier = SklearnClassifier(scikit_classifier,sparse=False)
        elif(isinstance(scikit_classifier,GaussianNB)):
            self.classifier = SklearnClassifier(scikit_classifier,sparse=False)
        else:
            self.classifier = SklearnClassifier(scikit_classifier)
        self.compiled_templates = self.process_template(template_file_name)

        iob_data = file_to_instances(train_file_name)

        if(isinstance(scikit_classifier, CRF)):
            feature_sets = []

            logger.info("Training with %i instances " % len(iob_data))
            logger.info("Training with %i tokens" % count_tokens(iob_data))

            for n, instance in enumerate(iob_data):
                sentence_n = n
                pos_tags = [('z_POS', token[1]) for token in instance]
                labels = [token[2] for token in instance]
                tokens = [token[0] for token in instance]
                sentence_feats = []

                for n, token in enumerate(tokens):
                    dict_features = fe.get_features(
                            [token],
                            labels=labels,
                            outp_label=False,
                            legacy_features=pos_tags
                    )[0]
                    sentence_feats.append([dict_features, labels[n]])

                feature_sets.append(self.apply_feature_template(sentence_feats, out_label=True))

            X_train = [
                        [token[0] for token in instance]
                        for instance in feature_sets
                    ]
            y_train = [
                        [token[1] for token in instance]
                        for instance in feature_sets
                    ]
            self.classifier._clf = self.classifier._clf.fit(X_train, y_train)
            return

        else:
            feature_sets = []

            logger.info("Training with %i instances " % len(iob_data))
            logger.info("Training with %i tokens" % count_tokens(iob_data))
            for n, instance in enumerate(iob_data):
                sentence_n = n
                pos_tags = [('z_POS', token[1]) for token in instance]
                labels = [token[2] for token in instance]
                tokens = [token[0] for token in instance]
                for n,token in enumerate(tokens):
                    dict_features = fe.get_features([token],labels=labels,outp_label=False,legacy_features=pos_tags)[0]
                    feature_sets.append([dict_features, labels[n]])
            self.classifier.train(self.apply_feature_template(feature_sets,out_label=True))
            return

    def classify(self, feature_sets):
        """
        Args:
            feature_sets:
                a list of dictionaries like the following:

                [{'a_token': u'Nella',
                 'b_punct': 'OTHERS',
                 'c_brackets': 'OTHERS',
                 'd_case': 'INIT_CAPS',
                 'e_number': 'NO_DIGITS',
                 'f_ngram_1': u'N',
                 'f_ngram_2': u'Ne',
                 'f_ngram_3': u'Nel',
                 'f_ngram_4': u'Nell',
                 'g_ngram_1': u'a',
                 'g_ngram_2': u'la',
                 'g_ngram_3': u'lla',
                 'g_ngram_4': u'ella',
                 'h_lowcase': u'nella',
                 'i_str-length': '5',
                 'l_pattern': 'Aaaaa',
                 'm_compressed-pattern': 'Aa',
                 'n_works_dictionary': 'OTHERS',
                 'z': '_'}, ... ]

        Returns:
            result:
                a list of dictionaries, where each dictionary corresponds
                to a token,

                [{'features': [],
                 'id': 37,
                 'label': 'O',
                 'probs': {'B-AAUTHOR':
                     {'alpha': '234.113833',
                     'beta': '-2.125040',
                     'prob': '0.000262'},
                   },
                 'token': '.'},...]
        """
        # apply feature templates (from CRF++)
        template_feature_sets = self.apply_feature_template(
                                    feature_sets,
                                    out_label=False
                                )

        if(isinstance(self.classifier._clf, CRF)):
            output_labels = self.classifier._clf.predict(
                                [template_feature_sets]
                            )[0]
        else:
            # keep the output labels
            output_labels = self.classifier.classify_many(
                                template_feature_sets
                            )

        result = []
        for n,feature_set in enumerate(feature_sets):
            temp = {}
            temp["token"]=feature_set["a_token"].encode('utf-8')
            temp["label"]=str(output_labels[n])
            result.append(temp)
        return result
    def process_template(self,template_file):
        """

        Example of the output:

        [('U01:%s', [(-2, 0)]),
         ('U02:%s', [(-1, 0)]),...]

        """
        f = open(template_file,'r')
        lines = [line.replace('\n','') for line in f.readlines() if not line.startswith('\n') and not line.startswith('#') and not line.startswith('B')]
        f.close()
        import re
        exp = re.compile("%x\[(-?\d+),(-?\d+)\]")
        result = []
        for line in lines:
            result.append((exp.sub('%s',line),[(int(match[0]),int(match[1])) for match in exp.findall(line)]))
        return result
    def apply_feature_template(self,feature_sets,out_label=False):
        """
        TODO: apply each of the compiled templates
        """
        def get_value(feature_sets,token_n,feature_n):
            if(token_n < 0):
                return "ND"
            elif(token_n > (len(feature_sets)-1)):
                return "ND"
            else:
                return feature_sets[token_n][feature_n]
        if(out_label):
            unlabelled_feature_sets = [[f[0][key] for key in sorted(f[0])] for f in feature_sets]
        else:
            unlabelled_feature_sets = [[f[key] for key in sorted(f)] for f in feature_sets]
        assert len(feature_sets) == len(unlabelled_feature_sets)
        new_features = []
        for n,fs in enumerate(unlabelled_feature_sets):
            result = {}
            for template,replacements in self.compiled_templates:
                template_name = template.split(":")[0]
                template = template.split(":")[1]
                values = [get_value(unlabelled_feature_sets,n+r[0],r[1]) for r in replacements]
                result[template_name] = template%tuple(values)
            logger.debug("Feature set after applying template: %s" % result)
            if(out_label):
                # keep the expected label for training
                new_features.append([result,feature_sets[n][1]])
            else:
                new_features.append(result)
        return new_features

def chain_IOB_files(directories,output_fname,extension=".iob"):
    import glob
    import codecs
    all_in_one = []
    for dir in directories:
        # get all .iob files
        # concatenate their content with line return
        # write to a new file
        logger.debug("Processing %s"%dir)
        for infile in glob.glob( os.path.join(dir, '*%s'%extension) ):
            logger.debug("Found the file %s"%infile)
            file_content = codecs.open("%s"%(infile), 'r',encoding="utf-8").read()
            all_in_one.append(file_content)
    result = "\n\n".join(all_in_one)
    try:
        file = codecs.open(output_fname, 'w',encoding="utf-8")
        file.write(result)
        file.close()
        return True
    except Exception, e:
        raise e

class citation_extractor:
    """
    A Canonical Citation Extractor.
    First off, import the settings via module import
    >>> from settings import base_settings

    Then create an extractor passing as argument the settings file
    >>> extractor = citation_extractor(base_settings)

    Let's now get some test instances...
    >>> test = read_iob_files(base_settings.TEST_DIRS[0])

    We pass the postags and tokens separately:
    >>> postags = [[("z_POS",token[1]) for token in instance] for instance in test if len(instance)>0]
    >>> instances = [[token[0] for token in instance] for instance in test if len(instance)>0]

    And finally we classify the test instances
    >>> result = extractor.extract(instances, postags)

    As deafult, a CRF model is used. However, when initialising the `citation_extractor` you can
    pass on to it any scikit classifier, e.g. RandomForest:

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> extractor = citation_extractor(base_settings,RandomForestClassifier())

    """

    def __init__(self,options,classifier=None,labelled_feature_sets=None):
        self.classifier=None
        self.fe = FeatureExtractor()
        if(options.DATA_FILE != ""):
            allinone_iob_file = options.DATA_FILE
        elif(options.DATA_DIRS != ""):
            chain_IOB_files(options.DATA_DIRS,"%sall_in_one.iob"%options.TEMP_DIR,".txt")
            allinone_iob_file = "%sall_in_one.iob"%options.TEMP_DIR
        # initialise the classifier
        if(classifier is None):
            self.classifier=CRFPP_Classifier(allinone_iob_file,"%s%s"%(options.CRFPP_TEMPLATE_DIR,options.CRFPP_TEMPLATE),options.TEMP_DIR)
        else:
            self.classifier = ScikitClassifierAdapter(classifier,allinone_iob_file,"%s%s"%(options.CRFPP_TEMPLATE_DIR,options.CRFPP_TEMPLATE),labelled_feature_sets)

    def output(self,result,outp=None):
        """
        Outputs the result of extraction/classification in several formats.
        """
        fname = determine_path()+"/data/"+"temp.xml"
        f = open(fname,"w")
        temp = verbose_to_XML(result)
        f.write(temp)
        f.close()
        if(outp=="xml"):
            return temp
        elif(outp=="html"):
            import codecs
            fp = codecs.open(fname, "r", "utf-8")
            text = fp.read()
            fp.close()
            return out_html(text).decode("utf-8")
        elif(outp=="json"):
            return json.dumps(result)

    def extract(self, instances,legacy_features=None):
        """
        Extracts canonical citations from a list of instances, such as sentences or other meaningful and
        comparable subvisions of a text. This method acts as a proxy for the classify() method of the classifier.

        Args:
            instances: A list of instances, for example sentences.
        Returns:
            TODO describe
        """
        result = []
        for n,instance in enumerate(instances):
            if(legacy_features is not None):
                feat_sets = self.fe.get_features(instances[n],[],False,legacy_features[n])
            else:
                feat_sets = self.fe.get_features(instances[n],[],False)
            result.append(self.classifier.classify(feat_sets))
        return result

class FeatureExtractor:
    """
    A feature extractor to extract features from tokens.

    Usage:
        >>> fe = FeatureExtractor()
    """

    def __init__(self):
        self.OTHERS=0
        # brackets
        self.PAIRED_ROUND_BRACKETS=1
        self.UNPAIRED_ROUND_BRACKETS=2
        self.PAIRED_SQUARE_BRACKETS=3
        self.UNPAIRED_SQUARE_BRACKETS=4
        # case
        self.MIXED_CAPS=5
        self.ALL_CAPS=6
        self.INIT_CAPS=7
        self.ALL_LOWER=8
        # punctuation
        self.FINAL_DOT=10
        self.CONTINUING_PUNCTUATION=11
        self.STOPPING_PUNCTUATION=12
        self.QUOTATION_MARK=13
        self.HAS_HYPHEN=14
        self.NO_PUNCTUATION=15
        # number
        self.YEAR=16
        self.RANGE=17
        self.DOT_SEPARATED_NUMBER=18
        self.DOT_SEPARATED_PLUS_RANGE=19
        self.NUMBER=20
        self.ROMAN_NUMBER=21
        self.NO_DIGITS=9
        self.MIXED_ALPHANUM=22
        # dictionaries
        self.MATCH_AUTHORS_DICT=23
        self.MATCH_WORKS_DICT=24
        self.CONTAINED_AUTHORS_DICT=25
        self.CONTAINED_WORKS_DICT=26
        # misc

        self.feat_labels=['i']*30
        self.feat_labels[self.OTHERS]="OTHERS"
        # brackets
        self.feat_labels[self.PAIRED_ROUND_BRACKETS]="PAIRED_ROUND_BRACKETS"
        self.feat_labels[self.UNPAIRED_ROUND_BRACKETS]="UNPAIRED_ROUND_BRACKETS"
        self.feat_labels[self.PAIRED_SQUARE_BRACKETS]="PAIRED_SQUARE_BRACKETS"
        self.feat_labels[self.UNPAIRED_SQUARE_BRACKETS]="UNPAIRED_SQUARE_BRACKETS"
        # case
        self.feat_labels[self.MIXED_CAPS]="MIXED_CAPS"
        self.feat_labels[self.ALL_CAPS]="ALL_CAPS"
        self.feat_labels[self.INIT_CAPS]="INIT_CAPS"
        self.feat_labels[self.ALL_LOWER]="ALL_LOWER"
        # punctuation
        self.feat_labels[self.FINAL_DOT]="FINAL_DOT"
        self.feat_labels[self.CONTINUING_PUNCTUATION]="CONTINUING_PUNCTUATION"
        self.feat_labels[self.STOPPING_PUNCTUATION]="STOPPING_PUNCTUATION"
        self.feat_labels[self.QUOTATION_MARK]="QUOTATION_MARK"
        self.feat_labels[self.HAS_HYPHEN]="HAS_HYPHEN"
        self.feat_labels[self.NO_PUNCTUATION]="NO_PUNCTUATION"
        # number
        self.feat_labels[self.NO_DIGITS]="NO_DIGITS"
        self.feat_labels[self.YEAR]="YEAR"
        self.feat_labels[self.RANGE]="RANGE"
        self.feat_labels[self.DOT_SEPARATED_NUMBER]="DOT_SEPARATED_NUMBER"
        self.feat_labels[self.DOT_SEPARATED_PLUS_RANGE]="DOT_SEPARATED_PLUS_RANGE"
        self.feat_labels[self.NUMBER]="NUMBER"
        self.feat_labels[self.ROMAN_NUMBER]="ROMAN_NUMBER"
        self.feat_labels[self.MIXED_ALPHANUM]="MIXED_ALPHANUM"
        # dictionaries
        self.feat_labels[self.MATCH_AUTHORS_DICT]="MATCH_AUTHORS_DICT"
        self.feat_labels[self.MATCH_WORKS_DICT]="MATCH_WORKS_DICT"
        self.feat_labels[self.CONTAINED_AUTHORS_DICT]="CONTAINED_AUTHORS_DICT"
        self.feat_labels[self.CONTAINED_WORKS_DICT]="CONTAINED_WORKS_DICT"
        # dictionary matching

        self.init_dictionaries()

    def init_dictionaries(self):
        from citation_extractor.Utils.FastDict import LookupDictionary
        import codecs
        try:
            # initialise works dictionary
            fname = dir="%s/data/works.csv"%determine_path()
            file = codecs.open(fname,"r","utf-8")
            raw_data = file.read()
            file.close()
            self.works_dict = LookupDictionary(raw_data.encode('utf-8'))
        except Exception, e:
            raise e

        try:
            # initialise authors dictionary
            fname = dir="%s/data/authors.csv"%determine_path()
            file = codecs.open(fname,"r","utf-8")
            raw_data = file.read()
            file.close()
            self.authors_dict = LookupDictionary(raw_data.encode('utf-8'))
        except Exception, e:
            raise e
        return

    def extract_bracket_feature(self,check_str):
        """
        Extract a feature concerning the eventual presence of brackets

        Args:
            check_str: the string for which we need to extract features

        Returns:
            a tuple:
                result[0] : is the name of the feature
                result[1] : is the feature value expressed as integer

        Example:
            >>> tests = [u'(one)',u'another']
            >>> fe = FeatureExtractor()
            >>> [(tests[n],fe.feat_labels[fe.extract_bracket_feature(t)[1]]) for n,t in enumerate(tests)]
        """
        res = None
        # define check regexps
        pair_sq_bra=re.compile(r'\[.*?\]')
        unpair_sq_bra=re.compile(r'[\[\]]')
        pair_rd_bra=re.compile(r'\(.*?\)')
        unpair_rd_bra=re.compile(r'[\(\)]')
        # execute checks
        if(pair_sq_bra.search(check_str)):
            res = self.PAIRED_SQUARE_BRACKETS
        elif(unpair_sq_bra.search(check_str)):
            res = self.UNPAIRED_SQUARE_BRACKETS
        elif(pair_rd_bra.search(check_str)):
            res = self.PAIRED_ROUND_BRACKETS
        elif(unpair_rd_bra.search(check_str)):
            res = self.UNPAIRED_ROUND_BRACKETS
        else:
            res = self.OTHERS
        return ("c_brackets",res)

    def extract_case_feature(self,check_str):
        """
        Extracts a feature concerning the ortographic case of a token.

        Args:
            check_str: the string from which the feature will be extracted.
        Returns:
            A tuple TODO -> explain
        """
        naked = re.sub('[%s]' % re.escape(string.punctuation), '', check_str)
        res = self.OTHERS
        if(naked.isalpha()):
            if(naked.isupper()):
                res = self.ALL_CAPS
            elif(naked.islower()):
                res = self.ALL_LOWER
            elif(naked[0].isupper()):
                res = self.INIT_CAPS
        return ("d_case",res)

    def extract_punctuation_feature(self,check_str):
        """
        Checks the presence of hyphen and quotation marks.

        Args:
            check_str: the string for which we need to extract features

        Returns:
            a tuple:
                result[0] : is the name of the feature
                result[1] : is the feature value expressed as integer

        Example:
            >>> tests = [u'"',u'Iliad',u'"']
            >>> fe = FeatureExtractor()
            >>> [(tests[n],fe.feat_labels[fe.extract_punctuation_feature(t)[1]]) for n,t in enumerate(tests)]
            >>> tests = [u'«',u'De',u'uirginitate',u'»']
            >>> [(tests[n],fe.feat_labels[fe.extract_punctuation_feature(t)[1]]) for n,t in enumerate(tests)]

        """
        res = self.OTHERS
        punct_exp=re.compile('[%s]' % re.escape(string.punctuation))
        final_dot=re.compile(r'.*?\.$')
        three_dots=re.compile(r'.*?\.\.\.$')
        cont_punct=re.compile(r'.*?[,;:]$')
        quot_punct=re.compile(r'.*?[\"\'«»]')
        if(three_dots.match(check_str)):
            res = self.OTHERS
        elif(final_dot.match(check_str)):
            res = self.FINAL_DOT
        elif(cont_punct.match(check_str)):
            res = self.CONTINUING_PUNCTUATION
        elif(quot_punct.match(check_str)):
            res = self.QUOTATION_MARK
        #elif(punct_exp.match(check_str)):
            #res = self.OTHER_PUNCTUATION
        return ("b_punct",res)

    def extract_number_feature(self,check_str):
        """
        TODO
        1. first part of the features concerns the whole string
        2. second part should relate to the presence of number in a string
        * presence of range
        * presence of modern dates
        * is an ordinale number (Roman)?

        Example:
            >>> tests = [u'100',u'1994',u'1990-1999',u'23s.',u'10-11']
            >>> fe = FeatureExtractor()
            >>> [(tests[n],fe.feat_labels[fe.extract_number_feature(t)[1]]) for n,t in enumerate(tests)]
        """
        res = self.OTHERS
        naked = re.sub('[%s]' % re.escape(string.punctuation), '', check_str).lower()
        is_modern_date_range = r"(\d{4}-\d{4})"

        if(naked.isdigit()):
            res = self.NUMBER
        elif(naked.isalpha()):
            res = self.NO_DIGITS
        elif(naked.isalnum()):
            res = self.MIXED_ALPHANUM
        return ("e_number",res)

    def extract_char_ngrams(self,inp,size=4):
        """
        Extract ngram features (prefixes and suffixes), provided that the input string has a minimum length

        Args:
            inp: the string for which we need to extract features

        Returns:
            a list of tuples. each tuple:
                result[0] : is the name of the feature
                result[1] : is the feature value, in this case a string

        Example:
            >>> tests = [u'Hom',u'Esiodo',u'a']
            >>> fe = FeatureExtractor()
            >>> [fe.extract_char_ngrams(t) for t in tests]
        """
        out=[]
        nd="ND"
        inp  = u"%s"%inp
        for i in range(0,4): # ngram prefixes
            i+=1
            if(len(inp) >= size): # string length matches minimum size
                temp = ("f_ngram_%i"%i,inp[0:i])
            else:
                #  string length below minimum size
                temp = ("f_ngram_%i"%i,nd)
            out.append(temp)
        for i in range(0,4): # ngram suffixes
            i+=1
            if(len(inp) >= size):  # string length matches minimum size
                temp = ("g_ngram_%i"%(i),inp[len(inp)-i:])
            else:
                #  string length below minimum size
                temp = ("g_ngram_%i"%i,nd)
            out.append(temp)
        return out

    def extract_string_features(self,check_str):
        """
        Extract string length and text only string lowercase
        """
        out = re.sub('[%s]' % re.escape(string.punctuation), '', check_str)
        res = []
        if(not out==""):
            t = ('h_lowcase',out.lower())
            res.append(t)
            t = ('i_str-length',str(len(out)))
            res.append(t)
        else:
            t = ('h_lowcase','_')
            res.append(t)
            t = ('i_str-length',str(len(out)))
            res.append(t)
        res.append(('a_token',check_str))
        return res

    def extract_dictionary_feature(self, check_str):
        """
        TODO
        * check that the string is actually a word (may not be necessary with different tokenisation)

        Example:
            >>> tests = [u'Hom.',u'Homér']
            >>> fe = FeatureExtractor()
            >>> [(tests[n],fe.feat_labels[fe.extract_dictionary_feature(t)[1]]) for n,t in enumerate(tests)]

        """
        feature_name = "n_works_dictionary"

        # compile a list of stopwords for all relevant languages
        languages = ["it", "de", "fr", "en", "es"]
        stopwords = []
        for lang in languages:
            stopwords += safe_get_stop_words(lang)

        if len(check_str) <= 2 or check_str.lower() in stopwords:
            # don't output dictionary feature for stopwords!
            return (feature_name, self.OTHERS)

        match_works = self.works_dict.lookup(check_str.encode("utf-8"))
        match_authors = self.authors_dict.lookup(check_str.encode("utf-8"))
        #result = (feature_name, self.OTHERS)

        if(len(match_authors) > 0):
            for key in match_authors:
                if(len(match_authors[key]) == len(check_str)):
                    result = (feature_name, self.MATCH_AUTHORS_DICT)
                else:
                    result = (feature_name, self.CONTAINED_AUTHORS_DICT)
        elif(len(match_works) > 0):
            for key in match_works:
                if(len(match_works[key]) == len(check_str)):
                    result = (feature_name, self.MATCH_WORKS_DICT)
                else:
                    result = (feature_name, self.CONTAINED_WORKS_DICT)
        else:
            result = (feature_name, self.OTHERS)
        return result

    def extract_word_length_feature(self,check_str,threshold=5):
        """
        Features which gets fired when len(check_str) > threshold.
        TODO We should probably calculate (periodically) the average length for diff tags (aauthor,awork,refauwork).
        """
        pass

    def extract_pattern_feature(self,check_str):
        """
        >>> fe = FeatureExtractor()
        >>> test = u"Homéro,1999"
        >>> value = fe.extract_pattern_feature(test)
        >>> print value[1]
        Aaaaaa-0000
        """
        result=[]
        for n,char in enumerate(check_str):
            if(char.isalnum()):
                if(char.isalpha()):
                    if(char.islower()):
                        result.append('a')
                    else:
                        result.append('A')
                else:
                    result.append('0')
            else:
                result.append('-')
        return ('l_pattern',"".join(result))

    def extract_compressed_pattern_feature(self,check_str):
        """
        >>> fe = FeatureExtractor()
        >>> test = u"Homéro,1999"
        >>> value = fe.extract_compressed_pattern_feature(test)
        >>> print value[1]
        Aa-0
        """
        result=[]
        for n,char in enumerate(check_str):
            if(char.isalnum()):
                if(char.isalpha()):
                    if(char.islower()):
                        if(n+1 <= len(check_str)-1 and check_str[n+1].islower()):
                            pass
                        else:
                            result.append('a')
                    else:
                        if(n+1 <= len(check_str)-1 and check_str[n+1].isupper()):
                            pass
                        else:
                            result.append('A')
                else:
                    if(n+1 <= len(check_str)-1 and check_str[n+1].isdigit()):
                        pass
                    else:
                        result.append('0')
            else:
                if(n+1 <= len(check_str)-1 and (check_str[n+1].isalnum() is False)):
                    pass
                else:
                    result.append('-')
        return ('m_compressed-pattern',"".join(result))

    def extract_features(self,inp):
        feature_set=[]
        feat_funcs=[self.extract_punctuation_feature
        ,self.extract_bracket_feature
        ,self.extract_case_feature
        ,self.extract_number_feature
        ,self.extract_char_ngrams
        ,self.extract_string_features
        ,self.extract_pattern_feature
        ,self.extract_compressed_pattern_feature
        ,self.extract_dictionary_feature]
        for f in feat_funcs:
            result = f(inp)
            if(type(result) == types.TupleType):
                feature_set.append(result)
            elif(type(result) == types.ListType):
                for r in result:
                    feature_set.append(r)
        return feature_set

    def get_features(self,instance,labels=[],outp_label=True, legacy_features=None):
        """
        Args:
            instance:
                the instance to be classified, represented as a list of tokens.
            labels:
                ...
            outp_label:
                ...

        Example:
            >>> fe = FeatureExtractor()
            >>> test = ['cf.', 'Hom', 'Il.', '1.1', ';']
            >>> postags = [('z_POS','N/A'),('z_POS','N/A'),('z_POS','N/A'),('z_POS','N/A'),('z_POS','N/A')]
            >>> tmp = fe.get_features(test,outp_label=False,legacy_features=postags)
        """
        features = [self.extract_features(tok) for tok in instance]
        tok1 = features[0]
        keys = [f[0] for f in tok1]
        res = [dict(r) for r in features]
        logger = logging.getLogger('CREX.FeatureExtractor')
        for n,x in enumerate(res):
            # transform the numeric values into strings
            for key in keys:
                if(type(x[key]) is type(12)):
                    x[key] = self.feat_labels[x[key]] # get the string label corresponding to a given int value
                    #x[key] = str(x[key]) # leave the numeric feature value
            if(outp_label is True):
                x['z_gt_label']=labels[n]
        if(legacy_features is not None):
            for n,token in enumerate(res):
                token[legacy_features[n][0]] = legacy_features[n][1]
        logger.debug("\n"+"\n".join(instance_to_string(res)))
        return res

    def get_feature_order(self):
        """
        Returns the order in which the features are output.

        Example:
            >>> fe = FeatureExtractor()
            >>> fe.get_feature_order()
        """
        dumb_tok = ("Test.","O")
        temp = self.get_features([dumb_tok[0]],[dumb_tok[1]])[0]
        return [k for k in sorted(temp.keys())]

    def prepare_for_training(self,file_name):
        """
        #TODO: can be made staticmethod at some point

        Args:
            file_name: the input file in IOB format

        Returns:
            TODO document

        Example:
            >>> fe = FeatureExtractor() #doctest: +SKIP
            >>> print fe.prepare_for_training("data/75-02637.iob") #doctest: +SKIP
        """
        import codecs
        fp = codecs.open(file_name, "r", "utf-8")
        comment=re.compile(r'#.*?')
        lines = fp.read()
        instances=[group.split('\n')for group in lines.split("\n\n")]
        all_tokens = []
        all_labels = []
        all_postags = []
        for inst in instances:
            labels= []
            tokens=[]
            postags=[]
            for line in inst:
                if(not comment.match(line)):
                    temp = line.split('\t')
                    if(len(temp) == 2):
                        tokens.append(temp[0])
                        labels.append(temp[1])
                    else:
                        tokens.append(temp[0])
                        labels.append(temp[2])
                        postags.append(temp[1])
            all_labels.append(labels)
            all_tokens.append(tokens)
            if(len(postags) > 0):
                all_postags.append(postags)
        if(len(all_postags) > 0):
            all_postags = [[("z_POS",token) for token in instance] for instance in all_postags]
            res2 = [self.get_features(r,all_labels[n],legacy_features=all_postags[n]) for n,r in enumerate(all_tokens)]
        else:
            res2 = [self.get_features(r,all_labels[n]) for n,r in enumerate(all_tokens)]
        # all this fuss is to have instances and feature sets as text
        res2 = [instance_to_string(r) for r in res2]
        res3 = ["\n".join(i) for i in res2]
        out = "\n\n".join(res3)
        return out

if __name__ == "__main__":
    import doctest
    doctest.testmod()
