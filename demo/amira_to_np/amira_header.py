# -*- coding: utf-8 -*-
# amira_header.py
"""
Module to convert parsed data from an Amira header into a set of nested objects. The key class is :py:class:``AmiraHeader``.

Usage:

::

    >>> from amira.header import AmiraHeader
    >>> ah = AmiraHeader.from_file('file.am')
    >>> print(ah)

Each nested object is constructed from the :py:class:``Block`` class defined.

There are four top-level attributes that every ``AmiraHeader`` will have:

*    designation

*    definitions

*    parameters

*    data_pointers

Each attribute can be queried using the ``attrs`` attribute.

::

    >>> print(ah.data_pointers.attrs)
    ['data_pointer_1', 'data_pointer_2', 'data_pointer_3', 'data_pointer_4', 'data_pointer_5', 'data_pointer_6']
    >>> print(ah.data_pointers.data_pointer_1)
    data_pointer_1
    pointer_name: VERTEX
    data_format: None
    data_dimension: 3
    data_type: float
    data_name: VertexCoordinates
    data_index: 1
    data_length: None

Data pointers are identified by the name ``data_pointer_<n>``.

"""
import sys
import pprint

from .amira_grammar import get_parsed_header

class Block(object):
    """Generic block to be loaded with attributes"""
    def __init__(self, name):
        self.name = name
        self.attrs = list()

    def add_attr(self, name, value):
        """Add an attribute to an ``Block`` object"""
        setattr(self, str(name), value)
        self.attrs.append(str(name))

    def __str__(self):
        string = "{}\n".format(self.name)
        for attr in self.attrs:
            if isinstance(getattr(self, str(attr)), Block):
                string += "{}\n".format(getattr(self, str(attr)))
            else:
                string += "{}: {}\n".format(attr, getattr(self, str(attr)))
        return string

    @property
    def ids(self):
        """Convenience method to get the ids for Materials present"""
        assert self.name == "Materials"
        ids = list()
        for attr in self.attrs:
            attr_obj = getattr(self, attr)
            if hasattr(attr_obj, 'Id'):
                ids.append(getattr(attr_obj, 'Id'))
        return ids

    def __getitem__(self, index):
        """Convenience method to get an attribute with 'Id' for a Material"""
        assert self.name == "Materials"
        assert isinstance(index, int)
        for attr in self.attrs:
            attr_obj = getattr(self, attr)
            if hasattr(attr_obj, 'Id'):
                if getattr(attr_obj, 'Id') == index:
                    return attr_obj
                else:
                    continue # next attr
        else:
            return None


class AmiraHeader(object):
    """Class to encapsulate Amira metadata"""
    def __init__(self, raw_data, parsed_data):
        self._raw_data = raw_data
        self._parsed_data = parsed_data
        self._load()

    @classmethod
    def from_file(cls, fn, *args, **kwargs):
        """Constructor to build an AmiraHeader object from a file

        :param str fn: Amira file
        :return ah: object of class ``AmiraHeader`` containing header metadata
        :rtype: ah: :py:class:`ahds.header.AmiraHeader`
        """
        raw, parsed  = get_parsed_header(fn, *args, **kwargs)
        return AmiraHeader(raw, parsed)

    @classmethod
    def from_str(cls, raw, *args, **kwargs):
        """Constructor to build an AmiraHeader object from a string

        :param str raw: Amira raw header
        :return ah: object of class ``AmiraHeader`` containing header metadata
        :rtype: ah: :py:class:`ahds.header.AmiraHeader`
        """
        raw, parsed  = get_parsed_header(raw, *args, **kwargs)
        return AmiraHeader(raw, parsed)

    @property
    def raw_header(self):
        """Show the raw header data"""
        return self._raw_data

    @property
    def parsed_header(self):
        """Show the raw header data"""
        return self._parsed_data

    def _load(self):
        self._load_designation(self._parsed_data['designation'])
        self._load_definitions(self._parsed_data['definitions'])
        self._load_data_pointers(self._parsed_data['data_pointers'])
        # self._load_parameters(self._parsed_data['parameters'])

    @property
    def designation(self):
        """Designation of the Amira file defined in the first row

        Designations consist of some or all of the following data:

        *    filetype e.g. ``AmiraMesh`` or ``Avizo`` or``HyperSurface``

        *    dimensions e.g. ``3D``

        *    format e.g. ``BINARY-LITTLE-ENDIAN``

        *    version e.g. ``2.1``

        *    extra format e.g. ``<hxsurface>``
        """
        return self._designation

    @property
    def definitions(self):
        """Definitions consist of a key-value pair specified just after the
        designation preceded by the key-word 'define'
        """
        return self._definitions

    @property
    def parameters(self):
        """The set of parameters for each of the segments specified
        e.g. colour, data pointer etc.
        """
        return self._parameters

    @property
    def data_pointers(self):
        """The list of data pointers together with a name, data type, dimension,
        index, format and length
        """
        return self._data_pointers

    def _load_designation(self, block_data):
        self._designation = Block('designation')
        if 'filetype' in block_data:
            self._designation.add_attr('filetype', block_data['filetype'])
        else:
            self._designation.add_attr('filetype', None)
        if 'dimension' in block_data:
            self._designation.add_attr('dimension', block_data['dimension'])
        else:
            self._designation.add_attr('dimension', None)
        if 'format' in block_data:
            self._designation.add_attr('format', block_data['format'])
        else:
            self._designation.add_attr('format', None)
        if 'version' in block_data:
            self._designation.add_attr('version', block_data['version'])
        else:
            self._designation.add_attr('version', None)
        if 'extra_format' in block_data:
            self._designation.add_attr('extra_format', block_data['extra_format'])
        else:
            self._designation.add_attr('extra_format', None)

    def _load_definitions(self, block_data):
        self._definitions = Block('definitions')
        for key in block_data:
            self._definitions.add_attr(key, block_data[key])

    def _load_parameters(self, block_data):
        self._parameters = Block('parameters')
        for parameter in block_data:
            if 'nested_parameter' in parameter:
                nested_parameter = parameter['nested_parameter']
                self._parameters.add_attr(nested_parameter['nested_parameter_name'], Block(nested_parameter['nested_parameter_name']))
                nested_parameter_obj = getattr(self._parameters, str(nested_parameter['nested_parameter_name']))
                for nested_parameter_value in nested_parameter['nested_parameter_values']:
                    if 'attributes' in nested_parameter_value:
                        if nested_parameter_value['attributes']:
                            nested_parameter_obj.add_attr(nested_parameter_value['name'], Block(nested_parameter_value['name']))
                            nested_parameter_value_obj = getattr(nested_parameter_obj, str(nested_parameter_value['name']))
                            for attribute in nested_parameter_value['attributes']:
                                nested_parameter_value_obj.add_attr(attribute['attribute_name'], attribute['attribute_value'])
                        else:
                            nested_parameter_obj.add_attr(nested_parameter_value['name'], None)
                    elif 'nested_attributes' in nested_parameter_value:
                        nested_parameter_obj.add_attr(nested_parameter_value['name'], Block(nested_parameter_value['name']))
                        for nested_attribute in nested_parameter_value['nested_attributes']:
                            nested_attribute_obj = getattr(nested_parameter_obj, str(nested_parameter_value['name']))
                            nested_attribute_obj.add_attr(nested_attribute['nested_attribute_name'], Block(nested_attribute['nested_attribute_name']))
                            nested_attribute_value_obj = getattr(nested_attribute_obj, str(nested_attribute['nested_attribute_name']))
                            for nested_attribute_value in nested_attribute['nested_attribute_values']:
                                nested_attribute_value_obj.add_attr(nested_attribute_value['nested_attribute_value_name'], nested_attribute_value['nested_attribute_value_value'])
                    else:
                        nested_parameter_obj.add_attr(nested_parameter_value['name'], nested_parameter_value['inline_parameter_value'])
            if 'inline_parameter' in parameter:
                inline_parameter = parameter['inline_parameter']
                self._parameters.add_attr(inline_parameter['inline_parameter_name'], inline_parameter['inline_parameter_value'])

    def _load_data_pointers(self, block_data):
        self._data_pointers = Block('data_pointers')
        for data_index in block_data:
            data_pointer_name = "data_pointer_{}".format(data_index)
            self._data_pointers.add_attr(data_pointer_name, Block(data_pointer_name))
            pointer_obj = getattr(self._data_pointers, data_pointer_name)
            data_pointer = block_data[data_index]

            if 'pointer_name' in data_pointer:
                pointer_obj.add_attr('pointer_name', data_pointer['pointer_name'])
            else:
                pointer_obj.add_attr('pointer_name', None)
            if 'data_format' in data_pointer:
                pointer_obj.add_attr('data_format', data_pointer['data_format'])
            else:
                pointer_obj.add_attr('data_format', None)
            if 'data_dimension' in data_pointer:
                pointer_obj.add_attr('data_dimension', data_pointer['data_dimension'])
            else:
                pointer_obj.add_attr('data_dimension', None)
            if 'data_type' in data_pointer:
                pointer_obj.add_attr('data_type', data_pointer['data_type'])
            else:
                pointer_obj.add_attr('data_type', None)
            if 'data_name' in data_pointer:
                pointer_obj.add_attr('data_name', data_pointer['data_name'])
            else:
                pointer_obj.add_attr('data_name', None)
            if 'data_index' in data_pointer:
                pointer_obj.add_attr('data_index', data_pointer['data_index'])
            else:
                pointer_obj.add_attr('data_index', None)
            if 'data_length' in data_pointer:
                pointer_obj.add_attr('data_length', data_pointer['data_length'])
            else:
                pointer_obj.add_attr('data_length', None)

    def __repr__(self):
        return "<AmiraHeader with {:,} bytes>".format(len(self))

    def __str__(self):
        string = "*" * 50 + "\n"
        string += "AMIRA HEADER\n"
        string += "-" * 50 + "\n"
        string += "{}\n".format(self.designation)
        string += "-" * 50 + "\n"
        string += "{}\n".format(self.definitions)
        string += "-" * 50 + "\n"
        string += "{}\n".format(self.parameters)
        string += "-" * 50 + "\n"
        string += "{}\n".format(self.data_pointers)
        string += "*" * 50
        return string
