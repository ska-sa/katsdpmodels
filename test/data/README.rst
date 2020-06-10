Test models
===========

The files in this directory are made available to the tests via a web server.

Working models
--------------
rfi_mask_ranges.hdf5
    RFI mask with two ranges, one with finite and one with infinite
    max_baseline_length.

Aliases
-------
direct.alias
    Refers directly to a valid model.
indirect.alias
    Refers to direct.alias.
loop.alias
    Refers to itself, making an infinite loop.

Broken models
-------------
bad_model_type.hdf5
    Contains a ``model_type`` field that is not legal.
no_model_type.hdf5
    The ``model_type`` field is absent.
not_hdf5.hdf5
    Not a valid HDF5 file
rfi_mask_bad_format.hdf5
    The ``model_format`` field is not a valid value.
rfi_mask_missing_dataset.hdf5
    The ``ranges`` dataset is absent.
rfi_mask_ranges_2d.hdf5
    The ``ranges`` dataset has 2 dimensions instead of 1.
rfi_mask_ranges_is_group.hdf5
    There is a group called ``ranges`` rather than a dataset.
wrong_extension.blah
    Copy of rfi_mask_ranges.hdf5 with a different extension.
