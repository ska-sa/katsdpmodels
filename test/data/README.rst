Test models
===========

The files in this directory are made available to the tests via a web server.

Working models
--------------
rfi_mask_ranges.h5
    RFI mask with two ranges, one with finite and one with infinite
    max_baseline_length.
rfi_mask_ranges_metadata.h5
    Same as above with the addition of model_comment, model_author etc.

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
bad_model_type.h5
    Contains a ``model_type`` field that is not legal.
no_model_type.h5
    The ``model_type`` field is absent.
bad_model_version.h5
    Contains a ``model_version`` that is a string instead of an int.
no_model_version.h5
    The ``model_version`` field is absent.
not_hdf5.h5
    Not a valid HDF5 file
rfi_mask_bad_format.h5
    The ``model_format`` field is not a valid value.
rfi_mask_missing_dataset.h5
    The ``ranges`` dataset is absent.
rfi_mask_ranges_2d.h5
    The ``ranges`` dataset has 2 dimensions instead of 1.
rfi_mask_ranges_is_group.h5
    There is a group called ``ranges`` rather than a dataset.
wrong_extension.blah
    Copy of rfi_mask_ranges.h5 with a different extension.

Other
-----
all_bytes.bin
    Byte values running from 0 to 255.
