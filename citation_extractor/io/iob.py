"""
Functions to deal with input/output of data in CONLL/IOB format.
"""

from __future__ import print_function
import logging
import codecs
import glob
import os
import re

logger = logging.getLogger(__name__)


def count_tokens(instances):
    """Short summary.

    :param type instances: Description of parameter `instances`.
    :return: Description of returned object.
    :rtype: type

    """
    return sum([1 for instance in instances for token in instance])


def write_iob_file(instances, dest_file):
    """Write a set of instances to an IOB file.

    :param list instances: Description of parameter `instances`.
    :param str dest_file: Description of parameter `dest_file`.
    :return: Description of returned object.
    :rtype: boolean

    """
    to_write = "\n\n".join(
        [
            "\n".join(["\t".join(token) for token in instance])
            for instance in instances
        ]
    )
    try:
        with codecs.open(dest_file, 'w', 'utf-8') as f:
            f.write(to_write)
        return True
    except Exception, e:
        raise e


def file_to_instances(inp_file):
    """Reads a IOB file a converts it into a list of instances.

    :param type inp_file: Path to the input IOB file.
    :return: A list of tuples, where tuple[0] is the token and tuple[1]
        contains its assigned label.
    :rtype: list

    """
    f = codecs.open(inp_file,"r","utf-8")
    inp_text = f.read()
    f.close()
    out=[]
    comment=re.compile(r'#.*?')
    for i in inp_text.split("\n\n"):
        inst=[]
        for j in i.split("\n"):
            if(not comment.match(j)):
                temp = tuple(j.split("\t"))
                if(len(temp)>1):
                    inst.append(temp)
        if(inst and len(inst)>1):
            out.append(inst)
    return out


def instance_contains_label(instance, labels=["O"]):
    """
    TODO:
    """
    temp=[token[len(token)-1] for token in instance]
    res = set(temp).intersection(set(labels))
    if(len(res)==0):
        return False
    else:
        return True


def filter_IOB(instances, tag_name):
    """docstring for filter_IOB"""
    out=[]
    res=[]
    temp = []
    count = 0
    for instance in instances:
        temp = []
        open = False
        for i in instance:
                if(i[2]=='B-%s'%tag_name):
                    temp.append(i[0])
                    open = True
                elif(i[2]=='I-%s'%tag_name):
                    if(open):
                        temp.append(i[0])
                elif(i[2]=='O'):
                    if(open):
                        out.append(temp)
                        temp = []
                        open = False
                    else:
                        pass
        if(len(temp) > 0 and open):
            out.append(temp)
    for r in out:
        res.append(' '.join(r))
    return res


def read_iob_files(inp_dir, extension=".iob"):
    instances = []
    for infile in glob.glob( os.path.join(inp_dir, '*%s'%extension) ):
        temp = file_to_instances(infile)
        instances +=temp
        logger.debug("read %i instances from file %s"%(len(temp),infile))
    return instances


def instance_to_string(instance):
    """Converts a feature dictionary into a string representation.

    :param type instance: Description of parameter `instance`.
    :return: Description of returned object.
    :rtype: type

    """
    def token_to_string(tok_dict):
        return [tok_dict[k] for k in sorted(tok_dict.keys())]

    return [
        "\t".join(token_to_string(feature_set))
        for feature_set in instance
    ]
