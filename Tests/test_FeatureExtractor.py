from citation_extractor.core import FeatureExtractor as FE
import pprint

class TestExample:
    def test_c(self):
        assert 'c' == 'c'
	def test_b(self):
		assert 'b' == 'b'
		
class TestEFeatureExtractor:
    def test(self):
		fe = FE()
		inp = "Hesiod is a Greek poet".split(" ")
		res = fe.new_extract_features(inp,outp_label=False)
		out = [dict(r) for r in res]
		pprint.pprint(out)