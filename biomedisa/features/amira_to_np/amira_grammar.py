# -*- coding: utf-8 -*-
"""
Grammar to parse headers in Amira files
"""

import sys
import re
import time
import collections

from pprint import pprint

TOKEN_NAME = 'name'
TOKEN_NUMBER = 'number'
TOKEN_STRING = 'string'
TOKEN_OP = 'op'
TOKEN_COMMENT = 'comment'
TOKEN_NEWLINE = 'newline'
TOKEN_COMMA = 'comma'
TOKEN_COLON = 'colon'
TOKEN_EQUALS = 'equals'
TOKEN_ENDMARKER = 'endmarker'
TOKEN_BYTEDATA_INFO = 'bytedata_info'

class Matcher:
    def __init__(self,rexp):
        self.rexp = rexp
    def __call__(self, buf ):
        matchobj = self.rexp.match( buf )
        return matchobj is not None

re_file_info = re.compile(r'^#(\s*)(AmiraMesh|HyperSurface|Avizo)(\s*)(?:3D)?(\s+)(BINARY-LITTLE-ENDIAN|BINARY|ASCII)(\s+)(\d+\.\d*)$')
is_file_info = Matcher(re_file_info)

re_string_literal = re.compile(r'^".*"$')
is_string_literal = Matcher(re_string_literal)

re_bytedata_info = re.compile(r'^(Lattice)(\s+)\{(\s*)(\w+)(\s+)(\w+)(\s*)\}(\s+)@(\d+)(\((\w+),(\d+)\))?$')
is_bytedata_info = Matcher(re_bytedata_info)

re_float = re.compile(r'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')
is_number = Matcher(re_float)

re_name = re.compile(r'^[a-zA-Z0-9_&:-]+(\[\d\])?$')
is_name = Matcher(re_name)

re_quoted_whitespace_splitter = re.compile(r'(".*")|[ \t\n]')

def lim_repr(value):
    full = repr(value)
    if len(full) > 100:
        full = full[:97]+'...'
    return full

class Tokenizer:
    def __init__( self, str ):
        self.buf = str
        self.last_tokens = []
        self._designation = {}
        self._data_pointers = {}
        self._definitions = {}
        self._parameters = {}

    def add_defines(self, define_dict ):
        self._definitions.update(define_dict)
    def add_parameters(self, dict ):
        self._parameters.update(dict)
    def get_tokens( self ):
        # keep a running accumulation of last 2 tokens
        for token_enum,token in enumerate(self._get_tokens()):
            self.last_tokens.append( token )
            while len(self.last_tokens) > 3:
                self.last_tokens.pop(0)
            if token_enum==0:
                if token[0] == TOKEN_COMMENT:
                    matchobj = re_file_info.match( token[1] )
                    if matchobj is not None:
                        items = list(filter(lambda x : x is not None and len(x.strip())>0, matchobj.groups()))
                        if "3D" in items:
                            self._designation =  {'filetype':items[0],
                                                  'dimension':items[1],
                                                  'format': items[-2],
                                                  'version':items[-1]}
                        else:
                            self._designation =  {'filetype':items[0],
                                                  'format': items[-2],
                                                  'version':items[-1]}
                else:
                    warnings.warn('Unknown file type. Parsing may fail.')
            yield token
    def _get_tokens( self ):
        newline = '\n' #b'\n'
        lineno = 0
        while len(self.buf) > 0:
            # get the next line -------
            idx = self.buf.index(newline)+1
            this_line, self.buf = self.buf[:idx], self.buf[idx:]
            lineno += 1

            # now parse the line into tokens ----
            if this_line.lstrip().startswith('#'):
                yield ( TOKEN_COMMENT, this_line[:-1], (lineno,0), (lineno, len(this_line)-1), this_line )
                yield ( TOKEN_NEWLINE, this_line[-1:], (lineno,len(this_line)-1), (lineno, len(this_line)), this_line )
            elif this_line==newline:
                yield ( TOKEN_NEWLINE, this_line, (lineno,0), (lineno, 1), this_line )
            elif is_bytedata_info( this_line ):
                matchobj = re_bytedata_info.match( this_line )
                items = list(filter(lambda x : x is not None and len(x.strip())>0, matchobj.groups()))
                assert(len(items)>=4)
                esdict = {'pointer_name':items[0],
                        'data_type':items[1],
                        'data_name':items[2],
                        'data_index': int(items[3])}
                if len(items)>4:
                    esdict['data_format'] = items[-2]
                    esdict['data_length'] = int(items[-1])
                self._data_pointers[items[3]]=esdict
                yield ( TOKEN_BYTEDATA_INFO, this_line, (lineno,0), (lineno, 1), this_line )
            else:
                parts = re_quoted_whitespace_splitter.split(this_line)
                parts.append(newline) # append newline
                parts = [p for p in parts if p is not None and len(p)]

                maybe_comma_part_idx = len(parts)-2 if len(parts) >= 2 else None

                colno = 0
                for part_idx, part in enumerate(parts):
                    startcol = colno
                    endcol = len(part)+startcol
                    colno = endcol + 1

                    if part_idx == maybe_comma_part_idx:
                        if len(part) > 1 and part.endswith(','):
                            # Remove the comma from further processing
                            part = part[:-1]
                            endcol -= 1
                            # Emit a comma token.
                            yield ( TOKEN_COMMA, part, (lineno,endcol), (lineno, endcol+1), this_line )
                    if part in ['{','}']:
                        yield ( TOKEN_OP, part, (lineno,startcol), (lineno, endcol), this_line )
                    elif part==newline:
                        yield ( TOKEN_NEWLINE, part, (lineno,startcol-1), (lineno, endcol-1), this_line )
                    elif part==':':
                        yield ( TOKEN_COLON, part, (lineno,startcol-1), (lineno, endcol-1), this_line )
                    elif part=='=':
                        yield ( TOKEN_EQUALS, part, (lineno,startcol-1), (lineno, endcol-1), this_line )
                    elif part==',':
                        yield ( TOKEN_COMMA, part, (lineno,startcol-1), (lineno, endcol-1), this_line )
                    elif is_number(part):
                        yield ( TOKEN_NUMBER, part, (lineno,startcol), (lineno, endcol), this_line )
                    elif is_name(part):
                        yield ( TOKEN_NAME, part, (lineno,startcol), (lineno, endcol), this_line )
                    elif is_string_literal(part):
                        yield ( TOKEN_STRING, part, (lineno,startcol), (lineno, endcol), this_line )
                    else:
                        raise NotImplementedError( 'cannot tokenize part %r (line %r)'%(lim_repr(part), lim_repr(this_line)) )
        yield ( TOKEN_ENDMARKER, '', (lineno,0), (lineno, 0), '' )

def atom( src, token, tokenizer, level=0, block_descent=False ):
    space = '  '*level
    end_block = None
    if token[0]==TOKEN_NAME:
        name = token[1]

        if block_descent:
            result = name
        else:
            next_token = next(src)

            if next_token[0] == TOKEN_OP and next_token[1]=='{':
                # this name begins a '{' block
                value, ended_with = atom( src, next_token, tokenizer, level=level+1 ) # create {}
                result = {name: value}

            else:
                # continue until newline
                elements = []
                ended_with = None
                force_colon = False
                while not (next_token[0] == TOKEN_NEWLINE):

                    if next_token[0] == TOKEN_COLON:
                        force_colon = True
                        next_token = next(src)

                    value, ended_with = atom( src, next_token, tokenizer, level=level+1, block_descent=force_colon ) # fill element of []
                    elements.append( value )
                    if ended_with is not None:
                        break
                    next_token = next(src)

                if ended_with is not None:
                    end_block = ended_with
                else:
                    # loop ended because we hit a newline
                    end_block = 'newline'

                elements = [e for e in elements if e is not None]
                if len(elements)==0:
                    result = name
                elif len(elements)==1:
                    result = {name: elements[0]}
                else:
                    result = {name: elements}
    elif token[0] in [TOKEN_COMMENT, TOKEN_COMMA]:
        result = None
    elif token[0] == TOKEN_OP and token[1]=='}':
        result = None
        end_block = 'block'
    elif token[0] == TOKEN_NEWLINE:
        result = None
        end_block = 'newline'
    elif token[0] == TOKEN_OP and token[1]=='{':
        if block_descent:
            raise RuntimeError('descent blocked but encountered block')

        elements = []

        # parse to end of block
        next_token = next(src)
        while not (next_token[0] == TOKEN_OP and next_token[1] == '}'):
            value, ended_with = atom( src, next_token, tokenizer, level=level+1 ) # fill element of {}
            elements.append( value )
            if ended_with=='block':
                break
            next_token = next(src)

        elements = [e for e in elements if e is not None]
        result = collections.OrderedDict()
        for element in elements:
            if isinstance(element,dict):
                for key in element:
                    assert key not in result
                    result[key] = element[key]
            else:
                assert isinstance(element,type(u'unicode string'))
                assert element not in result
                result[element] = None
    elif token[0]==TOKEN_NUMBER:
        try:
            value = int(token[1])
        except ValueError:
            value = float(token[1])
        result = value
    elif token[0]==TOKEN_STRING:
        value = token[1]
        result = value
    elif token[0]==TOKEN_BYTEDATA_INFO:
        result = None
    elif token[0]==TOKEN_EQUALS:
        result = None
    else:
        raise ValueError('unexpected token type: %r'%(token[0],))

    return result, end_block

def detect_format(fn, format_bytes=50, *args, **kwargs):
    """Detect Amira file format (AmiraMesh or HyperSurface)

    :param str fn: file name
    :param int format_bytes: number of bytes in which to search for the format [default: 50]
    :return str file_format: either ``AmiraMesh`` or ``HyperSurface``
    """
    assert format_bytes > 0

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

    return file_format


def get_header(fn, file_format, header_bytes=536870912, *args, **kwargs): #2097152
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
                m = re.search(b'(?P<data>)\n@1\n', rough_header, flags=re.S)
            elif file_format == "HyperSurface":
                m = re.search(b'(?P<data>)\nVertices [0-9]*\n', rough_header, flags=re.S)
            elif file_format == "Undefined":
                raise ValueError("Unable to parse undefined file")
        if m is None:
            header_count += header_bytes
        else:
            # select the data
            # data = m.group('data')
            idx = m.start()
            data = rough_header[:idx]
            return data

def parse_header(data, *args, **kwargs):
    """Parse the data using the grammar specified in this module

    :param str data: delimited data to be parsed for metadata
    :return list parsed_data: structured metadata
    """
    # the parser
    tokenizer = Tokenizer( data )
    src = tokenizer.get_tokens()

    token = next(src)

    while token[0] != TOKEN_ENDMARKER:
        this_atom, ended_with = atom(src, token, tokenizer) # get top-level atom
        if this_atom is not None:
            if isinstance( this_atom, dict ):
                if 'define' in this_atom:
                    tokenizer.add_defines( this_atom['define'] )
                else:
                    tokenizer.add_parameters( this_atom )
        token = next(src)

    result = {'designation' : tokenizer._designation,
            'definitions' : tokenizer._definitions,
            'data_pointers': tokenizer._data_pointers,
            'parameters': tokenizer._parameters}
    return result

def get_parsed_header(data, *args, **kwargs):
    """All above functions as a single function

    :param str fn: file name
    :return list parsed_data: structured metadata
    """
    assert(isinstance(data, str) or isinstance(data, bytes))

    # if data is str, it is a filename
    if isinstance(data, str):
        file_format = detect_format(data, *args, **kwargs)
        raw_header = get_header(data, file_format, *args, **kwargs)
    else: # otherwise data is raw header in bytes
        raw_header = data

    # remove parameters block
    p = raw_header.find(b'Parameters {')
    x = raw_header.find(b'Lattice {')
    header_wo_params = raw_header[:p] + raw_header[x:]

    # clean text header from reading as binary - remove b'', "\\n" to '\n'
    header_wo_params = str(header_wo_params).strip("b'").replace("\\n", '\n')

    # end in new line
    if header_wo_params[-1] != '\n':
        header_wo_params += '\n'

    parsed_data = parse_header(header_wo_params, *args, **kwargs)

    return raw_header, parsed_data
