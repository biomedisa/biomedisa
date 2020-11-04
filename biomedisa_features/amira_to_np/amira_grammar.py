# -*- coding: utf-8 -*-
"""
Grammar to parse headers in Amira files
"""

import sys
import re
from pprint import pprint
import time

# simpleparse
from simpleparse.parser import Parser
from simpleparse.common import numbers, strings  # @UnusedImport
from simpleparse.dispatchprocessor import DispatchProcessor, getString, dispatchList, dispatch, singleMap, multiMap  # @UnusedImport


class AmiraDispatchProcessor(DispatchProcessor):
    """Class defining methods to handle each token specified in the grammar"""
    def designation(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return {'designation': singleMap(taglist, self, buffer_)}
    def filetype(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def dimension(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def format(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def version(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def extra_format(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def comment(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return {'comment': singleMap(taglist, self, buffer_)}
    def date(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def definitions(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return {'definitions': dispatchList(self, taglist, buffer_)}
    def definition(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return singleMap(taglist, self, buffer_)
    def definition_name(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def definition_value(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        _av = dispatchList(self, taglist, buffer_)
        if len(_av) == 1:
            return _av[0]
        elif len(_av) > 1:
            return _av
        else:
            raise ValueError('definition value list is empty:', _av)
    def parameters(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return {'parameters': dispatchList(self, taglist, buffer_)}
    def parameter(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return singleMap(taglist, self, buffer_)
    def nested_parameter(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return singleMap(taglist, self, buffer_)
    def nested_parameter_name(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def nested_parameter_values(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return dispatchList(self, taglist, buffer_)
    def nested_parameter_value(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return singleMap(taglist, self, buffer_)
    def name(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def attributes(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        _a = dispatchList(self, taglist, buffer_)
        if _a:
            return _a
        else:
            return
    def attribute(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return singleMap(taglist, self, buffer_)
    def attribute_name(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def attribute_value(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        _av = dispatchList(self, taglist, buffer_)
        if len(_av) == 1:
            return _av[0]
        elif len(_av) > 1:
            return _av
        elif len(_av) == 0:
            return None
        else:
            raise ValueError('attribute value list is empty:', _av)
    def nested_attributes(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return dispatchList(self, taglist, buffer_)
    def nested_attribute(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return singleMap(taglist, self, buffer_)
    def nested_attribute_name(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def nested_attribute_values(self, tup, buffer_):  # @UnusedVariable
        tag, left, right, taglist = tup
        return dispatchList(self, taglist, buffer_)
    def nested_attribute_value(self, tup, buffer_):
        tag, left, right, taglist = tup
        return singleMap(taglist, self, buffer_)
    def nested_attribute_value_name(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def nested_attribute_value_value(self, tup, buffer_):
        tag, left, right, taglist = tup
        _av = dispatchList(self, taglist, buffer_)
        if len(_av) == 1:
            return _av[0]
        elif len(_av) > 1:
            return _av
        elif len(_av) == 0:
            return None
        else:
            raise ValueError('nested attribute value list is empty:', _av)
    def inline_parameter(self, tup, buffer_):
        tag, left, right, taglist = tup
        return singleMap(taglist, self, buffer_)
    def inline_parameter_name(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def inline_parameter_value(self, tup, buffer_):
        tag, left, right, taglist = tup
        if taglist[0][0] == "qstring":
            return getString((tag, left, right, taglist), buffer_)
        else:
            return dispatchList(self, taglist, buffer_)
    def data_pointers(self, tup, buffer_):
        tag, left, right, taglist = tup
        return {'data_pointers': dispatchList(self, taglist, buffer_)}
    def data_pointer(self, tup, buffer_):
        tag, left, right, taglist = tup
        return singleMap(taglist, self, buffer_)
    def pointer_name(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def data_type(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def data_dimension(self, tup, buffer_):
        tag, left, right, taglist = tup
        return int(getString((tag, left, right, taglist), buffer_))
    def data_name(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def data_index(self, tup, buffer_):
        tag, left, right, taglist = tup
        return int(getString((tag, left, right, taglist), buffer_))
    def data_format(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def data_length(self, tup, buffer_):
        tag, left, right, taglist = tup
        return int(getString((tag, left, right, taglist), buffer_))
    def hyphname(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def xstring(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def qstring(self, tup, buffer_):
        tag, left, right, taglist = tup
        return getString((tag, left, right, taglist), buffer_)
    def number(self, tup, buffer_):
        tag, left, right, taglist = tup
        if taglist[0][0] == 'int':
            return int(getString((tag, left, right, taglist), buffer_))
        elif taglist[0][0] == 'float':
            return float(getString((tag, left, right, taglist), buffer_))
        else:
            return getString((tag, left, right, taglist), buffer_)
    def number_seq(self, tup, buffer_):
        tag, left, right, taglist = tup
        return dispatchList(self, taglist, buffer_)


# Amira Header Grammar
amira_header_grammar = r'''
amira                        :=    designation, tsn, comment*, tsn*, definitions, tsn*, parameters*, tsn, data_pointers, tsn

designation                  :=    ("#", ts, filetype, ts, dimension*, ts*, format, ts, version, ts*, extra_format*, tsn) / ("#", ts, filetype, ts, version, ts, format, tsn)
filetype                     :=    "AmiraMesh" / "HyperSurface" / "Avizo"
dimension                    :=    "3D"
format                       :=    "BINARY-LITTLE-ENDIAN" / "BINARY" / "ASCII"
version                      :=    number
extra_format                 :=    "<", "hxsurface", ">"

comment                      :=    ts, ("#", ts, "CreationDate:", ts, date) / ("#", ts, xstring) , tsn
date                         :=    xstring

definitions                  :=    definition*
definition                   :=    ("define", ts, definition_name, ts, definition_value) / ("n", definition_name, ts, definition_value), tsn
definition_name              :=    hyphname
definition_value             :=    number, (ts, number)*

parameters                   :=    "Parameters", ts, "{", tsn, parameter*, "}", tsn
parameter                    :=    nested_parameter / inline_parameter / comment

nested_parameter             :=    ts, nested_parameter_name, ts, "{", tsn, nested_parameter_values, ts, "}", tsn
nested_parameter_name        :=    hyphname
nested_parameter_values      :=    nested_parameter_value*
nested_parameter_value       :=    ts, name, ts, ("{", tsn, attributes, ts, "}") / ("{", tsn, nested_attributes, ts, "}") / inline_parameter_value, tsn
name                         :=    hyphname

attributes                   :=    attribute*
attribute                    :=    ts, attribute_name, ts, attribute_value, c*, tsn
attribute_name               :=    hyphname
attribute_values             :=    attribute_value*
attribute_value              :=    ("-"*, "\""*, ((number, (ts, number)*) / xstring)*, "\""*)

nested_attributes            :=    nested_attribute*
nested_attribute             :=    ts, nested_attribute_name, ts, "{", tsn, nested_attribute_values, ts, "}", tsn
nested_attribute_name        :=    hyphname
nested_attribute_values      :=    nested_attribute_value*
nested_attribute_value       :=    ts, nested_attribute_value_name, ts, nested_attribute_value_value, c*, tsn
nested_attribute_value_name  :=    hyphname
nested_attribute_value_value :=    "-"*, "\""*, ((number, (ts, number)*) / xstring)*, "\""*

inline_parameter             :=    ts, inline_parameter_name, ts, inline_parameter_value, c*, tsn
inline_parameter_name        :=    hyphname
inline_parameter_value       :=    (number, (ts, number)*) / qstring, c*

data_pointers                :=    data_pointer*
data_pointer                 :=    pointer_name, ts, "{", ts, data_type, "["*, data_dimension*, "]"*, ts, data_name, ts, "}", ts, "="*, ts, "@", data_index, "("*, data_format*, ","*, data_length*, ")"*, tsn
pointer_name                 :=    hyphname
data_type                    :=    hyphname
data_dimension               :=    number
data_name                    :=    hyphname
data_index                   :=    number
data_format                  :=    "HxByteRLE" / "HxZip"
data_length                  :=    number

hyphname                     :=    [A-Za-z_], [A-Za-z0-9_\-]*
qstring                      :=    "\"", "["*, [A-Za-z0-9_,=.\(\|)(\):/ \t\n]*, "]"*, "\""
xstring                      :=    [A-Za-z], [A-Za-z0-9(\|)_\- (\xef)(\xbf)(\xbd)]*
number_seq                   :=    number, (ts, number)*

# silent production rules
<tsn>                        :=    [ \t\n]*
<ts>                         :=    [ \t]*
<c>                          :=    ","
'''


def detect_format(fn, format_bytes=50, verbose=False, *args, **kwargs):
    """Detect Amira file format (AmiraMesh or HyperSurface)

    :param str fn: file name
    :param int format_bytes: number of bytes in which to search for the format [default: 50]
    :param bool verbose: verbose (default) or not
    :return str file_format: either ``AmiraMesh`` or ``HyperSurface``
    """
    assert format_bytes > 0
    assert verbose in [True, False]

    with open(fn, 'rb') as f:
        rough_header = f.read(format_bytes)

        if re.match(r'.*AmiraMesh.*', str(rough_header)):
            file_format = "AmiraMesh"
        elif re.match(r'.*Avizo.*', str(rough_header)):
            file_format = "Avizo"
        elif re.match(r'.*HyperSurface.*', str(rough_header)):
            file_format = "HyperSurface"
        else:
            file_format = "Undefined"

    if verbose:
        print(sys.stderr,  "{} file detected...".format(file_format))

    return file_format


def get_header(fn, file_format, header_bytes=536870912, verbose=False, *args, **kwargs):
    """Apply rules for detecting the boundary of the header

    :param str fn: file name
    :param str file_format: either ``AmiraMesh`` or ``Avizo``or ``HyperSurface``
    :param int header_bytes: number of bytes in which to search for the header [default: 20000]
    :return str data: the header as per the ``file_format``
    """
    assert header_bytes > 0
    assert file_format in ['AmiraMesh', 'Avizo', 'HyperSurface']
    header_count = header_bytes

    while (True):
        m = None
        with open(fn, 'rb') as f:
            rough_header = f.read(header_count)
            if file_format == "AmiraMesh" or file_format == "Avizo":
                if verbose:
                    print(sys.stderr, "Using pattern: (?P<data>.*)\\n@1\\n")
                m = re.search(b'(?P<data>.*)\n@1\n', rough_header, flags=re.S)
            elif file_format == "HyperSurface":
                if verbose:
                    print(sys.stderr, "Using pattern: (?P<data>.*)\\nVertices [0-9]*\\n")
                m = re.search(b'(?P<data>.*)\nVertices [0-9]*\n', rough_header, flags=re.S)
            elif file_format == "Undefined":
                raise ValueError("Unable to parse undefined file")
        if m is None:
            header_count += header_bytes
        else:
            # select the data
            data = m.group('data')
            return data

def parse_header(data, verbose=False, *args, **kwargs):
    """Parse the data using the grammar specified in this module

    :param str data: delimited data to be parsed for metadata
    :return list parsed_data: structured metadata
    """
    # the parser
    if verbose:
        print(sys.stderr, "Creating parser object...")
    parser = Parser(amira_header_grammar)

    # the processor
    if verbose:
        print(sys.stderr, "Defining dispatch processor...")
    amira_processor = AmiraDispatchProcessor()

    # parsing
    if verbose:
        print(sys.stderr, "Parsing data...")
    success, parsed_data, next_item = parser.parse(data, production='amira', processor=amira_processor)

    if success:
        if verbose:
            print(sys.stderr, "Successfully parsed data...")
        return parsed_data
    else:
        raise TypeError("Parse: {}\nNext: {}\n".format(parsed_data, next_item))


def get_parsed_data(fn, *args, **kwargs):
    """All above functions as a single function

    :param str fn: file name
    :return list parsed_data: structured metadata
    """

    file_format = detect_format(fn, *args, **kwargs)
    raw_header = get_header(fn, file_format, *args, **kwargs)

    # remove parameters block
    p = raw_header.find(b'Parameters {')
    x = raw_header.find(b'Lattice {')
    header_wo_params = raw_header[:p] + raw_header[x:]

    # clean text header from reading as binary - remove b'', "\\n" to '\n'
    header_wo_params = str(header_wo_params).strip("b'").replace("\\n", '\n')

    parsed_data = parse_header(header_wo_params, *args, **kwargs)

    return raw_header, parsed_data

def get_parsed_data_str(raw_header, *args, **kwargs):
    """All above functions as a single function

    :param str raw_header: raw header
    :return list parsed_data: structured metadata
    """

    # remove parameters block
    p = raw_header.find(b'Parameters {')
    x = raw_header.find(b'Lattice {')
    header_wo_params = raw_header[:p] + raw_header[x:]

    # clean text header from reading as binary - remove b'', "\\n" to '\n'
    header_wo_params = str(header_wo_params).strip("b'").replace("\\n", '\n')

    parsed_data = parse_header(header_wo_params, *args, **kwargs)

    return raw_header, parsed_data
