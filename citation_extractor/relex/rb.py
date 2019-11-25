from operator import itemgetter
import logging

logger = logging.getLogger(__name__)


class RBRelationExtractor(object):

    def __init__(self):
        return

    def extract(self, document):

        relations = {}
        arg1 = None
        arg2 = None

        # why it's important to sort this way the entities?
        entities = list(document["entities"].values())
        entities.sort(key=itemgetter('start_offset'))

        for entity in entities:
            if(entity['entity_type'] != "REFSCOPE"):
                arg1 = entity['id']
                arg2 = None
            else:
                arg2 = entity['id']
                if(arg1 is not None):
                    rel_id = "R{}".format(len(list(relations.keys())) + 1)
                    relations[rel_id] = (arg1, arg2)
                    logger.debug(
                        "Detected relation{}".format(relations[rel_id])
                    )

        return relations
