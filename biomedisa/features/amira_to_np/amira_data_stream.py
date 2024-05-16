# -*- coding: utf-8 -*-
# data_stream

from collections import UserList
import numpy
import re
import struct
import time

from skimage.measure._find_contours import find_contours
from numba import jit

from .amira_header import AmiraHeader

# type of data to find in the stream
FIND = {
    'decimal': '\d',    # [0-9]
    'alphanum_': '\w',  # [a-aA-Z0-9_]
    }

def byterle_decoder(input_data, output_size):
    """Python drop-in replacement for compiled equivalent

    :param int output_size: the number of items when ``data`` is uncompressed
    :param str data: a raw stream of data to be unpacked
    :return numpy.array output: an array of ``numpy.uint8``
    """

    # input_data = struct.unpack('<{}B'.format(len(data)), data)
    # input_data = numpy.ndarray((len(data),), '<B', data)
    output = numpy.zeros(output_size, dtype=numpy.uint8)
    i = 0
    count = True
    repeat = False
    no = None
    j = 0
    len_data = len(input_data)
    while i < len_data:
        if count:
            no = input_data[i]
            if no > 127:
                no &= 0x7f # 2's complement
                count = False
                repeat = True
                i += 1
                continue
            else:
                i += 1
                count = False
                repeat = False
                continue
        elif not count:
            if repeat:
                value = input_data[i:i + no]
                repeat = False
                count = True
                output[j:j+no] = numpy.array(value)
                i += no
                j += no
                continue
            elif not repeat:
                value = input_data[i]
                output[j:j+no] = value
                i += 1
                j += no
                count = True
                repeat = False
                continue

    assert j == output_size
    return output

def byterle_encoder(data):
    if len(data) == 0:
        return ""

    base = 128
    max = 127

    output = []
    no_rep = []
    i = 0
    prev = None
    cur = None
    cnt = 0
    len_data = len(data)
    while i < len_data:
        cur = data[i]
        if prev == cur: # repeat
            if len(no_rep) > 0:
                output.append(base+len(no_rep))
                output.extend(no_rep)
                no_rep = []
            cnt = 1 # including prev
            while i < len_data:
                cur = data[i]
                if prev == cur:
                    cnt += 1
                    if cnt == max:
                        output.append(cnt)
                        output.append(cur)
                        prev = None
                        cur = None
                        cnt = 0
                        i += 1
                        break
                else:  # end of repeat
                    output.append(cnt)
                    output.append(prev)
                    prev = None
                    cnt = 0
                    break
                prev = cur
                i += 1
            if cnt > 0: # end of file
                output.append(cnt)
                output.append(cur)
                prev = None
                cur = None
                cnt = 0
        else: # no repeat
            if prev is not None:
                no_rep.append(prev)
                if len(no_rep) == max:
                    output.append(base+len(no_rep))
                    output.extend(no_rep)
                    no_rep = []
            prev = cur
            i += 1

    if cur is not None:
        no_rep.append(cur)

    if len(no_rep) > 0:
        output.append(base+len(no_rep))
        output.extend(no_rep)
        no_rep = []

    final_result = bytearray(output)
    return final_result

@jit(nopython=True)
def numba_array_create(dtype, base_size=1073741824):
    assert(base_size > 0)
    new_arr = numpy.zeros(base_size, dtype=dtype)
    return new_arr

@jit(nopython=True)
def numba_array_copy(arr):
    assert(arr.size > 0)
    new_arr = numpy.zeros(arr.size, dtype=arr.dtype)
    for i in range(arr.size):
        new_arr[i] = arr[i]
    return new_arr, new_arr.size

@jit(nopython=True)
def numba_array_resize(arr, base_size=1073741824):
    assert(base_size > 0)
    new_arr = numpy.zeros(arr.size+base_size, dtype=arr.dtype)
    new_arr[:arr.size] = arr
    return new_arr

@jit(nopython=True)
def numba_array_add(arr, idx, elem, base_size=1073741824):
    assert(base_size > 0)
    if idx >= arr.size:
        arr = numba_array_resize(arr, base_size)
    arr[idx] = elem
    idx += 1
    return arr, idx

@jit(nopython=True)
def numba_array_extend(arr, idx, elems, elems_len, base_size=1073741824):
    assert(base_size > 0)
    assert(elems.size <= base_size)
    if idx+elems.size >= arr.size:
        arr = numba_array_resize(arr, base_size)
    for i in range(elems_len):
        arr[idx+i] = elems[i]
    idx += elems_len
    return arr, idx

@jit(nopython=True)
def numba_array_extend_mult(arr, idx, elem, no_elem, base_size=1073741824):
    assert(base_size > 0)
    assert(no_elem <= base_size)
    if idx+no_elem >= arr.size:
        arr = numba_array_resize(arr, base_size)
    arr[idx:idx+no_elem] = elem
    idx += no_elem
    return arr, idx

@jit(nopython=True)
def byterle_decoder_njit(data, len_data, output_size):
    """Python drop-in replacement for compiled equivalent

    :param int output_size: the number of items when ``data`` is uncompressed
    :param str data: a raw stream of data to be unpacked
    :return numpy.array output: an array of ``numpy.uint8``
    """

    output = numba_array_create(dtype=numpy.uint8)
    output_idx = 0

    count = True
    no_repeat = False
    no = None
    i = 0
    while i < len_data:
        if count:
            no = data[i]
            if no > 127:
                no &= 0x7f # 2's complement
                no_repeat = True
            else:
                no_repeat = False

            i += 1
            count = False
            continue
        elif not count:
            if no_repeat:
                no_rep, no_rep_idx = numba_array_copy(data[i:i + no])
                output, output_idx = numba_array_extend(output, output_idx, no_rep, no_rep_idx)
                i += no
            elif not no_repeat:
                elem = data[i]
                output, output_idx = numba_array_extend_mult(output, output_idx, elem, no)
                i += 1

            count = True
            no_repeat = False
            continue

    output = output[:output_idx]

    assert output_idx == output_size
    return output

@jit(nopython=True)
def byterle_encoder_njit(data, len_data):

    BASE = 128
    MAX = 127

    output = numba_array_create(dtype=numpy.uint8)
    no_rep = numba_array_create(dtype=numpy.uint8, base_size=MAX)
    output_idx = 0
    no_rep_idx = 0

    prev = None
    cur = None
    cnt = 0
    i = 0
    while i < len_data:
        cur = data[i]
        if prev is not None and prev == cur: # repeat
            if no_rep_idx > 0:
                output, output_idx = numba_array_add(output, output_idx, BASE+no_rep_idx)
                output, output_idx = numba_array_extend(output, output_idx, no_rep, no_rep_idx)
                no_rep_idx = 0

            # find all repeat
            cnt = 1 # prev
            while i < len_data:
                cur = data[i]
                if prev == cur:
                    cnt += 1
                    if cnt == MAX:
                        output, output_idx = numba_array_add(output, output_idx, cnt)
                        output, output_idx = numba_array_add(output, output_idx, cur)
                        prev = None
                        cur = None
                        cnt = 0
                        i += 1
                        break

                else:  # end of repeat
                    output, output_idx = numba_array_add(output, output_idx, cnt)
                    output, output_idx = numba_array_add(output, output_idx, prev)
                    prev = None
                    cnt = 0
                    break

                prev = cur
                i += 1

            if cnt > 0: # end of file
                output, output_idx = numba_array_add(output, output_idx, cnt)
                output, output_idx = numba_array_add(output, output_idx, cur)
                prev = None
                cur = None
                cnt = 0

        else: # no repeat
            if prev is not None:
                no_rep, no_rep_idx = numba_array_add(no_rep, no_rep_idx, prev)
                if no_rep_idx == MAX:
                    output, output_idx = numba_array_add(output, output_idx, BASE+no_rep_idx)
                    output, output_idx = numba_array_extend(output, output_idx, no_rep, no_rep_idx)
                    no_rep_idx = 0

            prev = cur
            i += 1

    if cur is not None:
        no_rep, no_rep_idx = numba_array_add(no_rep, no_rep_idx, cur)

    if no_rep_idx > 0:
        output, output_idx = numba_array_add(output, output_idx, BASE+no_rep_idx)
        output, output_idx = numba_array_extend(output, output_idx, no_rep, no_rep_idx)
        no_rep_idx = 0

    output = output[:output_idx]
    return output

def hxbyterle_decode(output_size, data):
    """Decode HxRLE data stream

    If C-extension is not compiled it will use a (slower) Python equivalent

    :param int output_size: the number of items when ``data`` is uncompressed
    :param str data: a raw stream of data to be unpacked
    :return numpy.array output: an array of ``numpy.uint8``
    """

    # input_data = struct.unpack('<{}B'.format(len(data)), data)
    input_data = numpy.ndarray((len(data),), '<B', data)

    output = byterle_decoder_njit(input_data, len(input_data), output_size)

    assert len(output) == output_size
    return output

def hxbyterle_encode(data):
    """Encode HxRLE data

    :param numpy.array data: an array of ``numpy.uint8``
    :return str output: packed data stream
    """

    buf = data.tobytes()
    # buf = data.astype('<B').tostring()

    if len(buf) == 0:
        return ""

    # output = byterle_encoder(buf)
    output = byterle_encoder_njit(buf, len(buf))
    output = output.tolist()

    final_output = bytearray(output)
    return final_output


def hxzip_decode(data_size, data):
    """Decode HxZip data stream

    :param int data_size: the number of items when ``data`` is uncompressed
    :param str data: a raw stream of data to be unpacked
    :return numpy.array output: an array of ``numpy.uint8``
    """
    import zlib
    data_stream = zlib.decompress(data)
    output = numpy.array(struct.unpack('<{}B'.format(len(data_stream)), data_stream), dtype=numpy.uint8)
    # output = numpy.ndarray((data_size,), '<B', data)

    assert len(output) == data_size
    return output

def hxzip_encode(data):
    """Encode HxZip data stream

    :param numpy.array data: an array of ``numpy.uint8``
    :return str output: packed data stream
    """
    import zlib

    buf = data.tobytes()
    # buf = data.astype('<B').tostring()
    output = zlib.compress(buf)
    return output

def to_numpy_dtype(data_type):

    # assume little endian
    if data_type == "float":
        dtype = "<f"
    elif data_type == "int":
        dtype = "<i"
    elif data_type == "uint":
        dtype = "<I"
    elif data_type == "short":
        dtype = "<h"
    elif data_type == "ushort":
        dtype = "<H"
    elif data_type == "long":
        dtype = "<l"
    elif data_type == "ulong":
        dtype = "<L"
    elif data_type == "byte":
        # dtype = "<b"
        dtype = "<B"    # changed to unsigned char

    return dtype

def unpack_binary(data_pointer, definitions, data):
    """Unpack binary data using ``struct.unpack``

    :param data_pointer: metadata for the ``data_pointer`` attribute for this data stream
    :type data_pointer: ``ahds.header.Block``
    :param definitions: definitions specified in the header
    :type definitions: ``ahds.header.Block``
    :param bytes data: raw binary data to be unpacked
    :return tuple output: unpacked data
    """

    if data_pointer.data_dimension:
        data_dimension = data_pointer.data_dimension
    else:
        data_dimension = 1 # if data_dimension is None

    data_type = to_numpy_dtype(data_pointer.data_type)

    # get this streams size from the definitions
    x, y, z = definitions.Lattice
    data_length = x * y * z

    # output = numpy.array(struct.unpack('<' + '{}'.format(data_type) * data_length, data)) # assume little-endian
    output = numpy.ndarray((data_length,), data_type, data)
    output = output.reshape(data_length, data_dimension)
    return output

def pack_binary(data_pointer, definitions, data):

    data_type = to_numpy_dtype(data_pointer.data_type)

    output = data.astype(data_type).tostring()
    return output

def unpack_ascii(data):
    """Unpack ASCII data using string methods``

    :param data_pointer: metadata for the ``data_pointer`` attribute for this data stream
    :type data_pointer: ``ahds.header.Block``
    :param definitions: definitions specified in the header
    :type definitions: ``ahds.header.Block``
    :param bytes data: raw binary data to be unpacked
    :return list output: unpacked data
    """
    # string: split at newlines -> exclude last list item -> strip space from each
    numstrings = map(lambda s: s.strip(), data.split('\n')[:-1])

    # check if string is digit (integer); otherwise float
    if len(numstrings) == len(filter(lambda n: n.isdigit(), numstrings)):
        output = map(int, numstrings)
    else:
        output = map(float, numstrings)
    return output

def pack_ascii(data):
    return data.tobytes()

def encode_data(fmt, dp, defn, data):
    if fmt == "HxByteRLE":
        return hxbyterle_encode(data)
    elif fmt == "HxZip":
        return hxzip_encode(data)
    elif fmt == "ASCII":
        return pack_ascii(data)
    elif fmt is None: # try to pack data
        return pack_binary(dp, defn,  data)
    else:
        return None

def write_amira(fname, header, data):
    header = header.tobytes()   # convert to byte array
    header_obj = AmiraHeader.from_str(header)

    raw_header = header_obj.raw_header
    file_format = header_obj.designation.format
    data_attr = header_obj.data_pointers
    definitions = header_obj.definitions

    if file_format.find("BINARY") == -1:
        raise ValueError("write_amira: unsupported file format %r" % file_format)

    num_stream = len(data)
    raw_data = []
    for i in range(num_stream):

        dp = getattr(data_attr, "data_pointer_{}".format(i+1))
        encoding = getattr(dp, "data_format")
        size = getattr(dp, "data_length")
        if size is None:
            x, y, z = definitions.Lattice
            size = x * y * z

        raw_str = encode_data(encoding, dp, definitions, data[i])
        raw_data.append(raw_str)

        new_size = len(raw_str)
        if new_size != size and encoding is not None:   # update header for new size
            enc_size = '@'+str(i+1)+'('+encoding+','+str(size)+')'
            new_enc_size = '@'+str(i+1)+'('+encoding+','+str(new_size)+')'
            raw_header = raw_header.replace(enc_size.encode("utf-8"), new_enc_size.encode("utf-8"))

    with open(fname, 'wb') as f:
        f.write(raw_header)

        for i in range(num_stream):
            f.write(b"\n@%d\n" % (i+1))
            f.write(raw_data[i])
            f.write(b'\n')

class Image(object):
    """Encapsulates individual images"""
    def __init__(self, z, array):
        self.z = z
        self.__array = array
        self.__byte_values = set(self.__array.flatten().tolist())
    @property
    def byte_values(self):
        return self.__byte_values
    @property
    def array(self):
        """Accessor to underlying array data"""
        return self.__array
    def equalise(self):
        """Increase the dynamic range of the image"""
        multiplier = 255 // len(self.__byte_values)
        return self.__array * multiplier
    @property
    def as_contours(self):
        """A dictionary of lists of contours keyed by byte_value"""
        contours = dict()
        for byte_value in self.__byte_values:
            if byte_value == 0:
                continue
            mask = (self.__array == byte_value) * 255
            found_contours = find_contours(mask, 254, fully_connected='high') # a list of array
            contours[byte_value] = ContourSet(found_contours)
        return contours
    @property
    def as_segments(self):
        return {self.z: self.as_contours}
    def show(self):
        """Display the image"""
        with_matplotlib = True
        try:
            import matplotlib.pyplot as plt
        except RuntimeError:
            import skimage.io as io
            with_matplotlib = False

        if with_matplotlib:
            equalised_img = self.equalise()

            _, ax = plt.subplots()

            ax.imshow(equalised_img, cmap='gray')

            import random

            for contour_set in self.as_contours.itervalues():
                r, g, b = random.random(), random.random(), random.random()
                [ax.plot(contour[:,1], contour[:,0], linewidth=2, color=(r,g,b,1)) for contour in contour_set]

            ax.axis('image')
            ax.set_xticks([])
            ax.set_yticks([])

            plt.show()
        else:
            io.imshow(self.equalise())
            io.show()
    def __repr__(self):
        return "<Image with dimensions {}>".format(self.array.shape)
    def __str__(self):
        return "<Image with dimensions {}>".format(self.array.shape)


class ImageSet(UserList):
    """Encapsulation for set of ``Image`` objects"""
    def __getitem__(self, index):
        return Image(index, self.data[index])
    @property
    def segments(self):
        """A dictionary of lists of contours keyed by z-index"""
        segments = dict()
        for i in xrange(len(self)):
            image = self[i]
            for z, contour in image.as_segments.iteritems():
                for byte_value, contour_set in contour.iteritems():
                    if byte_value not in segments:
                        segments[byte_value] = dict()
                    if z not in segments[byte_value]:
                        segments[byte_value][z] = contour_set
                    else:
                        segments[byte_value][z] += contour_set

        return segments
    def __repr__(self):
        return "<ImageSet with {} images>".format(len(self))


class ContourSet(UserList):
    """Encapsulation for a set of ``Contour`` objects"""
    def __getitem__(self, index):
        return Contour(index, self.data[index])
    def __repr__(self):
        string = "{} with {} contours".format(self.__class__, len(self))
        return string


class Contour(object):
    """Encapsulates the array representing a contour"""
    def __init__(self, z, array):
        self.z = z
        self.__array = array
    def __len__(self):
        return len(self.__array)
    def __iter__(self):
        return iter(self.__array)
    @staticmethod
    def string_repr(self):
        string = "<Contour at z={} with {} points>".format(self.z, len(self))
        return string
    def __repr__(self):
        return self.string_repr(self)
    def __str__(self):
        return self.string_repr(self)


class AmiraDataStream(object):
    """Base class for all Amira DataStreams"""
    match = None
    regex = None
    bytes_per_datatype = 4
    dimension = 1
    datatype = None
    find_type = FIND['decimal']
    def __init__(self, amira_header, data_pointer, stream_data):
        self.__amira_header = amira_header
        self.__data_pointer = data_pointer
        self.__stream_data = stream_data
        self.__decoded_length = 0
    @property
    def header(self):
        """An :py:class:``ahds.header.AmiraHeader`` object"""
        return self.__amira_header
    @property
    def data_pointer(self):
        """The data pointer for this data stream"""
        return self.__data_pointer
    @property
    def stream_data(self):
        """All the raw data from the file"""
        return self.__stream_data
    @property
    def encoded_data(self):
        """Encoded raw data in this stream"""
        return None
    @property
    def decoded_data(self):
        """Decoded data for this stream"""
        return None
    @property
    def decoded_length(self):
        """The length of the decoded stream data in relevant units e.g. tuples, integers (not bytes)"""
        return self.__decoded_length
    @decoded_length.setter
    def decoded_length(self, value):
        self.__decoded_length = value
    def __repr__(self):
        return "{} object of {:,} bytes".format(self.__class__, len(self.stream_data))


class AmiraMeshDataStream(AmiraDataStream):
    """Class encapsulating an AmiraMesh/Avizo data stream"""
    last_stream = False
    match = b'stream'
    def __init__(self, *args, **kwargs):
        if self.last_stream:
            self.regex = b'\n@{}\n(?P<%s>.*)' % self.match
        else:
            self.regex = b'\n@{}\n(?P<%s>.*)\n@{}' % self.match
        super(AmiraMeshDataStream, self).__init__(*args, **kwargs)
        if hasattr(self.header.definitions, 'Lattice'):
            X, Y, Z = self.header.definitions.Lattice
            data_size = X * Y * Z
            self.decoded_length = data_size
        elif hasattr(self.header.definitions, 'Vertices'):
            self.decoded_length = None
        elif self.header.parameters.ContentType == "\"HxSpreadSheet\"":
            pass
        elif self.header.parameters.ContentType == "\"SurfaceField\",":
            pass
        else:
            raise ValueError("Unable to determine data size")
    @property
    def encoded_data(self):
        i = self.data_pointer.data_index
        regex = self.regex.replace(b'{}', b'%d')
        cnt = regex.count(b"%d")
        if cnt == 1:
            regex = regex % (i)
        elif cnt == 2:
            regex = regex % (i, i+1)
        else:
            return None
        m = re.search(regex, self.stream_data, flags=re.S)
        return m.group(str(self.match).strip("b'"))
    @property
    def decoded_data(self):
        if self.data_pointer.data_format == "HxByteRLE":
            return hxbyterle_decode(self.decoded_length, self.encoded_data)
        elif self.data_pointer.data_format == "HxZip":
            return hxzip_decode(self.decoded_length, self.encoded_data)
        elif self.header.designation.format == "ASCII":
            return unpack_ascii(self.encoded_data)
        elif self.data_pointer.data_format is None: # try to unpack data
            return unpack_binary(self.data_pointer, self.header.definitions, self.encoded_data)
        else:
            return None
    def get_format(self):
        return self.data_pointer.data_format
    def to_images(self):
        if hasattr(self.header.definitions, 'Lattice'):
            X, Y, Z = self.header.definitions.Lattice
        else:
            raise ValueError("Unable to determine data size")
        image_data = self.decoded_data.reshape(Z, Y, X)

        imgs = ImageSet(image_data[:])
        return imgs
    def to_volume(self):
        """Return a 3D volume of the data"""
        if hasattr(self.header.definitions, "Lattice"):
            X, Y, Z = self.header.definitions.Lattice
        else:
            raise ValueError("Unable to determine data size")

        volume = self.decoded_data.reshape(Z, Y, X)
        return volume


class AmiraHxSurfaceDataStream(AmiraDataStream):
    """Base class for all HyperSurface data streams that inherits from ``AmiraDataStream``"""
    def __init__(self, *args, **kwargs):
        self.regex = r"%s (?P<%s>%s+)\n" % (self.match, self.match.lower(), self.find_type)
        super(AmiraHxSurfaceDataStream, self).__init__(*args, **kwargs)
        self.__match = re.search(self.regex, self.stream_data)
        self.__name = None
        self.__count = None
        self.__start_offset = None
        self.__end_offset = None
    @property
    def name(self):
        return self.__name
    @name.setter
    def name(self, value):
        self.__name = value
    @property
    def count(self):
        return self.__count
    @count.setter
    def count(self, value):
        self.__count = value
    @property
    def start_offset(self):
        return self.__start_offset
    @start_offset.setter
    def start_offset(self, value):
        self.__start_offset = value
    @property
    def end_offset(self):
        return self.__end_offset
    @end_offset.setter
    def end_offset(self, value):
        self.__end_offset = value
    @property
    def match_object(self):
        return self.__match
    def __str__(self):
        return """\
            \r{} object
            \r\tname:         {}
            \r\tcount:        {}
            \r\tstart_offset: {}
            \r\tend_offset:   {}
            \r\tmatch_object: {}""".format(
                self.__class__,
                self.name,
                self.count,
                self.start_offset,
                self.end_offset,
                self.match_object,
                )


class VoidDataStream(AmiraHxSurfaceDataStream):
    def __init__(self, *args, **kwargs):
        super(VoidDataStream, self).__init__(*args, **kwargs)
    @property
    def encoded_data(self):
        return []
    @property
    def decoded_data(self):
        return []


class NamedDataStream(VoidDataStream):
    find_type = FIND['alphanum_']
    def __init__(self, *args, **kwargs):
        super(NamedDataStream, self).__init__(*args, **kwargs)
        self.name = self.match_object.group(self.match.lower())


class ValuedDataStream(VoidDataStream):
    def __init__(self, *args, **kwargs):
        super(ValuedDataStream, self).__init__(*args, **kwargs)
        self.count = int(self.match_object.group(self.match.lower()))


class LoadedDataStream(AmiraHxSurfaceDataStream):
    def __init__(self, *args, **kwargs):
        super(LoadedDataStream, self).__init__(*args, **kwargs)
        self.count = int(self.match_object.group(self.match.lower()))
        self.start_offset = self.match_object.end()
        self.end_offset = max([self.start_offset, self.start_offset + self.count * (self.bytes_per_datatype * self.dimension)])
    @property
    def encoded_data(self):
        return self.stream_data[self.start_offset:self.end_offset]
    @property
    def decoded_data(self):
        points = struct.unpack('>' + ((self.datatype * self.dimension) * self.count), self.encoded_data)
        x, y, z = (points[::3], points[1::3], points[2::3])
        return zip(x, y, z)


class VerticesDataStream(LoadedDataStream):
    match = "Vertices"
    datatype = 'f'
    dimension = 3


class NBranchingPointsDataStream(ValuedDataStream):
    match = "NBranchingPoints"


class NVerticesOnCurvesDataStream(ValuedDataStream):
    match = "NVerticesOnCurves"


class BoundaryCurvesDataStream(ValuedDataStream):
    match = "BoundaryCurves"


class PatchesInnerRegionDataStream(NamedDataStream):
    match = "InnerRegion"


class PatchesOuterRegionDataStream(NamedDataStream):
    match = "OuterRegion"


class PatchesBoundaryIDDataStream(ValuedDataStream):
    match = "BoundaryID"


class PatchesBranchingPointsDataStream(ValuedDataStream):
    match = "BranchingPoints"


class PatchesTrianglesDataStream(LoadedDataStream):
    match = "Triangles"
    datatype = 'i'
    dimension = 3


class PatchesDataStream(LoadedDataStream):
    match = "Patches"
    def __init__(self, *args, **kwargs):
        super(PatchesDataStream, self).__init__(*args, **kwargs)
        self.__patches = dict()
        for _ in xrange(self.count):
            # in order of appearance
            inner_region = PatchesInnerRegionDataStream(self.header, None, self.stream_data[self.start_offset:])
            outer_region = PatchesOuterRegionDataStream(self.header, None, self.stream_data[self.start_offset:])
            boundary_id = PatchesBoundaryIDDataStream(self.header, None, self.stream_data[self.start_offset:])
            branching_points = PatchesBranchingPointsDataStream(self.header, None, self.stream_data[self.start_offset:])
            triangles = PatchesTrianglesDataStream(self.header, None, self.stream_data[self.start_offset:])
            patch = {
                'InnerRegion':inner_region,
                'OuterRegion':outer_region,
                'BoundaryID':boundary_id,
                'BranchingPoints':branching_points,
                'Triangles':triangles,
                }
            if inner_region.name not in self.__patches:
                self.__patches[inner_region.name] = [patch]
            else:
                self.__patches[inner_region.name] += [patch]
            # start searching from the end of the last search
            self.start_offset = self.__patches[inner_region.name][-1]['Triangles'].end_offset
            self.end_offset = None
    def __iter__(self):
        return iter(self.__patches.keys())
    def __getitem__(self, index):
        return self.__patches[index]
    def __len__(self):
        return len(self.__patches)
    @property
    def encoded_data(self):
        return None
    @property
    def decoded_data(self):
        return None


class DataStreams(object):
    """Class to encapsulate all the above functionality"""
    def __init__(self, fn, header, *args, **kwargs):
        # private attrs
        self.__fn = fn  # property
        if header is None:
            self.__amira_header = AmiraHeader.from_file(fn) # property
        else:
            self.__amira_header = header
        self.__data_streams = dict()
        self.__filetype = None
        self.__stream_data = None
        self.__data_streams = self.__configure()
    def __configure(self):
        with open(self.__fn, 'rb') as f:
            self.__stream_data = f.read() #.strip(b'\n')
            if self.__amira_header.designation.filetype == "AmiraMesh" or self.__amira_header.designation.filetype == "Avizo":
                self.__filetype = self.__amira_header.designation.filetype
                i = 0
                while i < len(self.__amira_header.data_pointers.attrs) - 1: # refactor
                    data_pointer = getattr(self.__amira_header.data_pointers, 'data_pointer_{}'.format(i + 1))
                    self.__data_streams[i + 1] = AmiraMeshDataStream(self.__amira_header, data_pointer, self.__stream_data)
                    i += 1
                AmiraMeshDataStream.last_stream = True
                data_pointer = getattr(self.__amira_header.data_pointers, 'data_pointer_{}'. format(i + 1))
                self.__data_streams[i + 1] = AmiraMeshDataStream(self.__amira_header, data_pointer, self.__stream_data)
                # reset AmiraMeshDataStream.last_stream
                AmiraMeshDataStream.last_stream = False
            elif self.__amira_header.designation.filetype == "HyperSurface":
                self.__filetype = "HyperSurface"
                if self.__amira_header.designation.format == "BINARY":
                    self.__data_streams['Vertices'] = VerticesDataStream(self.__amira_header, None, self.__stream_data)
                    self.__data_streams['NBranchingPoints'] = NBranchingPointsDataStream(self.__amira_header, None, self.__stream_data)
                    self.__data_streams['NVerticesOnCurves'] = NVerticesOnCurvesDataStream(self.__amira_header, None, self.__stream_data)
                    self.__data_streams['BoundaryCurves'] = BoundaryCurvesDataStream(self.__amira_header, None, self.__stream_data)
                    self.__data_streams['Patches'] = PatchesDataStream(self.__amira_header, None, self.__stream_data)
                elif self.__amira_header.designation.format == "ASCII":
                    self.__data_streams['Vertices'] = VerticesDataStream(self.__amira_header, None, self.__stream_data)
        return self.__data_streams
    @property
    def file(self): return self.__fn
    @property
    def header(self): return self.__amira_header
    @property
    def stream_data(self): return self.__stream_data
    @property
    def filetype(self): return self.__filetype
    def __iter__(self):
        return iter(self.__data_streams.values())
    def __len__(self):
        return len(self.__data_streams)
    def __getitem__(self, key):
        return self.__data_streams[key]
    def __repr__(self):
        return "{} object with {} stream(s): {}".format(
            self.__class__,
            len(self),
            ", ".join(map(str, self.__data_streams.keys())),
            )
