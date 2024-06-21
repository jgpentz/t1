"""
Tools for reading and writing RF-related files.
"""

import bz2
import datetime
import hashlib
import json
import os
import re
import tempfile
import warnings
import zipfile
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.io
import xarray as xr
from scipy.interpolate import interp1d

from .conversions.db import db2v

# NOTE: The following is a bandaid for the np.typing.Arraylike until we are ready to update numpy
# from numpy.typing import ArrayLike
ArrayLike = Union[np.ndarray, float, int, Tuple, List]
# xr.set_options(keep_attrs=True)

__all__ = (
    "ff2frd",
    "frd_hash",
    "frd2ff",
    "frd2nf",
    "frd2ps",
    "frd2s",
    "frd2ts",
    "frd2xarray",
    "join_snp",
    "join_touchstone",
    "nf2frd",
    "ps2frd",
    "read_htf",
    "read_meta_data",
    "read_nf",
    "read_rs_bin",
    "read_touchstone",
    "s2frd",
    "s2xarray",
    "ts2frd",
    "validate_frd",
    "write_touchstone",
    "xarray2frd",
)

# These attributes are auto-populated and should not appear in user data
RESERVED_ATTRS = ["_source_url", "_source_hash"]

PARAMETERS = "S, Y, Z, H, G"
SI_SCALAR = {"hz": 1e0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}

# Regex matching floats with exponents
RE_FLOAT = re.compile(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?")
RE_COMMENT = re.compile(r"\s*!\s+(.*)")
RE_FORMAT = re.compile(
    r"#\s*(?P<freq_unit>mhz|ghz|khz|hz)\s+(?P<param>s|y|z|h|g)\s+(?P<format>db|ma|ri)(\s+r\s*(?P<z0>[0-9]+))?",
    flags=re.I,
)
RE_NOTDATA = re.compile(r".*[!#a-df-zA-DF-Z]+")

# TODO: add lots of .lower() for strings


def ff2frd(filepath, data, *args, compress: bool = False, version: Union[float, int] = 4, **kwargs) -> Path:
    data.attrs["type"] = "pattern"
    data.attrs["version"] = version
    return xarray2frd(filepath, data, compress=compress)


def frd2ff(*args, **kwargs):
    expected = "pattern"
    data = frd2xarray(*args, **kwargs)
    if data.attrs["type"] != expected:
        raise ValueError(f"Frd file type mismatch. Got '{data.attrs['type']}', expected '{expected}'")
    return data


def frd2nf(*args, **kwargs):
    expected = "nearfield"
    data = frd2xarray(*args, **kwargs)
    if data.attrs["type"] != expected:
        raise ValueError(f"Frd file type mismatch. Got '{data.attrs['type']}', expected '{expected}'")
    return data


def frd2s(*args, **kwargs):
    expected = "s-parameter"
    data = frd2xarray(*args, **kwargs)
    if data.attrs["type"] != expected:
        raise ValueError(f"Frd file type mismatch. Got '{data.attrs['type']}', expected '{expected}'")
    return data


def frd2ts(*args, **kwargs):
    expected = "timeseries"
    data = frd2xarray(*args, **kwargs)
    if data.attrs["type"] != expected:
        raise ValueError(f"Frd file type mismatch. Got '{data.attrs['type']}', expected '{expected}'")
    return data


def frd2ps(*args, **kwargs):
    expected = "power-sweep"
    data = frd2xarray(*args, **kwargs)
    if data.attrs["type"] != expected:
        raise ValueError(f"Frd file type mismatch. Got '{data.attrs['type']}', expected '{expected}'")
    return data


def frd2xarray(filepath: Union[str, PathLike, BytesIO], validate: bool = True, name: str = "noname") -> xr.DataArray:
    """Load an FRD file of any type. Depending on self-reported file type

    Parameters
    ----------
    filepath : str or Path or io.BytesIO
    validate : bool, optional
        If False, no checks will be performed to ensure file has correct fields for reported file type.
        Intended use is to read old FRD files which abused 's-parameter' type and did not store s-parameters.
    name : str, optional
        Name of the returned DataArray. Note: a name is required by functions such as merge and combine_by_coords which
        fail with the default None. This value is apparently ignored by most the xarray library.

    Returns
    -------
    DataArray

    """

    if isinstance(filepath, BytesIO):
        file_ = filepath
    elif Path(filepath).suffix.lower() in (".bz2",):
        file_ = bz2.open(filepath)
    else:
        file_ = open(filepath, mode="rb")

    # Compute hash and reset file_ reader
    # We read here to ensure hashes are based on the uncompressed file content
    source_hash = hashlib.md5(file_.read()).hexdigest()
    file_.seek(0)
    # Read the data for real
    d = scipy.io.loadmat(file_, squeeze_me=False)
    file_.close()

    def dict2xarray(d: dict) -> xr.DataArray:
        # begin parsing file
        keys = list(d.keys())

        # Check if we have a 1 dimension and convert it to a list
        # this avoids an error where extend will add each letter of the key
        # instead of adding the whole key as a single item
        if isinstance(d["dimensions"], str):
            d["dimensions"] = [d["dimensions"]]

        # same check if we only have one coord
        try:
            if isinstance(d["coord_keys"], str):
                coord_keys = [d["coord_keys"]]
            else:
                coord_keys = d["coord_keys"]
        except (KeyError, ValueError):
            # infer coord keys for pypattern rev4 files
            coord_keys = d["dimensions"]

        # handle dimensions singleton data
        if np.shape(d["dimensions"]) == (0, 0):
            dims = []
        else:
            dims = [str(dd[0]) for dd in d["dimensions"][0]]

        # handle coords for singleton data
        if np.shape(coord_keys) == (0, 0):
            coord_keys = []
        else:
            coord_keys = [str(k[0]) for k in coord_keys[0]]

        coords = {k: d[k] for k in coord_keys}

        coord_keys_physical = []
        for c in coords.keys():
            if coords[c].dtype.kind in ("U", "S"):
                # .mat file pads strings to equal length
                coords[c] = np.char.strip(coords[c])

            if c in dims:
                # dimensioned coordinate
                if coords[c].dtype.kind not in ("U", "S"):
                    # non-string arrays are 1xN since all matlab arrays have at least 2 dimensions
                    if coords[c].shape[0] != 1:
                        raise ValueError("Dimensioned coordinates must be 1D")
                    # Handle string coords eg keys.
                    # TODO: There is probably a better way to handle this!
                    if coords[c][0].dtype.kind == "O" and coords[c][0][0].dtype.kind in ("U", "S"):
                        # In this scenario coords[c][0] =
                        # array([array(['H_LIN_0'], dtype='<U7'), array(['V_LIN_0'], dtype='<U7')], dtype=object)
                        coords[c] = np.array([x[0] for x in coords[c][0]])
                    else:
                        coords[c] = coords[c][0]
            else:
                # non-dimensioned coordinate
                if coords[c].shape in ((1, 1), (1,)):
                    # singleton coordinate
                    coords[c] = coords[c].squeeze().item()
                elif coords[c].shape == (1, 2):
                    # physical coordinate
                    # handle these later after we know the shape of everything else
                    coord_keys_physical.append(c)
                else:
                    raise ValueError(f"Invalid format for coordinate '{c}'")

        values = np.reshape(d["values"], list(len(coords[d]) for d in dims))

        # revisit physical coordinates now that we've reshaped the data
        for c in coord_keys_physical:
            coord_dims = [str(cd.squeeze().item().strip()) for cd in coords[c][0, 0]]
            coord_values = coords[c][0, 1]
            coord_shape = [values.shape[list(dims).index(cc)] for cc in coord_dims]
            if coord_values.dtype.kind in ("U", "S"):
                coord_values = np.char.strip(coord_values)
            coord_values = np.reshape(coord_values, coord_shape)
            coords[c] = (tuple(coord_dims), coord_values)

        data_keys = {"values", "dimensions", "coord_keys", *coord_keys, *dims}
        common_attrs = ("document", "date", "type", "version")
        # Any remaining keys correspond to meta data
        meta_keys = set(keys) - set(data_keys) - set(common_attrs)

        # go back to list to maintain order
        # rip out any .mat stuff starting with "__" since we didn't set that
        attr_keys = [k for k in keys if (k in meta_keys and not str(k).startswith("__"))]

        def parse_dict(d: dict):
            for m in d.keys():
                try:
                    # try converting item to dict since mat files read as np structured arrays
                    if d[m].dtype.names is not None:
                        d[m] = dict(zip(d[m].dtype.names, list(x for x in d[m][0][0])))
                except AttributeError:
                    # this wasn't originally a dict
                    pass

                try:
                    # before any other modification we need to see if this is actually a nested xarray
                    d[m] = dict2xarray(d[m])
                    if validate:
                        validate_frd(d[m], defaults=False, error=False)

                except (AttributeError, KeyError):
                    if isinstance(d[m], dict):
                        d[m] = parse_dict(d[m])
                    else:
                        try:
                            # TODO: I think this and the reshaping below do the same stuff. Rip one out
                            d[m] = d[m].squeeze()
                            d[m] = d[m].item()
                        except ValueError:
                            # not singleton
                            pass

                        try:
                            # string fields
                            if d[m].dtype.kind in ("U", "S"):
                                # string arrays may be of length 0
                                d[m] = np.char.strip(d[m])
                                if d[m].shape == ():
                                    d[m] = d[m].item()
                                elif d[m].shape == (0,):
                                    d[m] = ""
                            elif np.shape(d[m]) in ((), (0,)):
                                d[m] = d[m].item()
                        except AttributeError:
                            # not an array. parsing is already done
                            pass
            return d

        attrs = {str(k): d[str(k)] for k in (*attr_keys, *common_attrs)}
        attrs = parse_dict(attrs)
        return xr.DataArray(values, dims=dims, coords=coords, attrs=attrs, name=name)

    da = dict2xarray(d)
    da.attrs["_source_url"] = str(filepath)
    da.attrs["_source_hash"] = source_hash

    if validate:
        validate_frd(da, defaults=False, error=False)

    return da


def _interp_snp(frequency: np.ndarray, *snp: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate multiple sNp datasets along common frequencies and stack.

    This method is used by read_touchstone to optionally interpolate
    overlapping data from frequency mismatched sNp files.

    Parameters
    ----------
    frequency :
        New frequency axis.
    snp :
        List of Numpy arraylike s-parameter data. Must have shape
        (frequency, ...) which restricts the usage of this function. This
        function is therfore only safely used within read_touchstone for now.
        If we used xarrays everywhere this would be easy to solve.

    Returns
    -------
    f : array_like
        Frequency matching `frequency.
    s : array_like
        Interpolated array of stacked snp data.
    """
    shape = [len(snp), len(frequency)] + list(snp[0][1].shape[1:])
    s_final = np.zeros((shape), dtype="complex")
    for ii, (x, s) in enumerate(snp):
        s_final[ii] = interp1d(x, s, axis=0)(frequency)

    return frequency, s_final


def join_snp(n: int, frequency: np.ndarray, snps: Tuple[tuple, np.ndarray], average_repeats=False) -> np.ndarray:
    """Combine multiple sNp matrices into a single, sNp structure.

    Parameters
    ----------
    n : int
        Number of ports.
    frequency : ndarray
        Frequency [Hz]
    snps : tuple of tuples
        A zipped set of port mappings and ndarrays with shape (F, n, n).

        Example::

            (
                ((1, 2, 3, ), p123_ndarray),
                ((1, 4, 5, ), p145_ndarray),
                ((2, 3, 5, ), p235_ndarray),
            )

    average_repeats : bool, optional
        Average repeated input and output match measurements.
        Not implemented!

    Returns
    -------
    s : array_like
        Combined s-parameters.
        Shape: (len(frequency), n, n)

    """
    # TODO: Add the repeat ability.

    frequency = np.atleast_1d(frequency)
    s_final = np.zeros((len(frequency), n, n), dtype="complex")

    for ii, (port_mapping, s) in enumerate(snps):
        port_mapping = np.atleast_1d(port_mapping)
        if s.shape != (frequency.shape[0], len(port_mapping), len(port_mapping)):
            raise ValueError(
                f"Data index {ii} must be of shape ({len(frequency)}, {port_mapping[0]}, {port_mapping[1]}) got "
                f"{s.shape}."
            )
        if np.any(port_mapping > (n)) or np.any(port_mapping <= 0):
            raise ValueError(f"Invalid mapping: '{port_mapping}'. Values must be between 1 and {n}.")

        # Using meshgrids, one can do non-contiguous slicing into a numpy array.
        mg_m, mg_n = np.meshgrid(port_mapping - 1, port_mapping - 1, indexing="ij")
        s_final[:, mg_m, mg_n] = s

    return s_final


def join_touchstone(
    n: int, snps: Tuple[tuple, Union[str, PathLike]], average_repeats=False
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine multiple sNp files into a single ndarray.

    These files must have *identical* independent vectors, otherwise the files
    will not be joined.

    Supports sparse data.

    Parameters
    ----------
    n : int
        Number of ports
    snps : tuple of tuples
        A zipped set of port mappings and filepaths. The length of the port mapping and the number of ports in the
        touchstone file must match!

        Example::

            (
                ((1, 2, ), 'my_file.s2p'),
                ((1, 3, ), 'my_file.s2p'),
                ((2, 3, ), 'my_file.s2p'),
            )

    average_repeats : bool, optional
        Average repeated input and output match measurements
        Not implemented!

    Returns
    -------
    x : array_like
        Independent variable for touchstone measurement.
        Shape: (len(frequency), )
    s : array_like
        Combined s-parameters.
        Shape: (len(frequency), n, n)

    """
    in_data = []

    # Find the length of x vector
    ref_file = Path(snps[0][1])
    ref_x, _ = read_touchstone(ref_file)

    for mapping, file_ in snps:
        x, snp = read_touchstone(file_)

        # Verify that all the frequency arrays are equal!
        try:
            np.testing.assert_array_almost_equal(ref_x, x)
        except AssertionError:
            raise ValueError(f'Mismatched frequency vectors in "{file_}" and "{ref_file}"')

        if snp.shape[-1] != len(mapping):
            raise ValueError(
                f"Dimension mismatch! Number of ports in {file_} does not match the provided mapping {mapping}."
            )

        in_data.append((mapping, snp))

    return ref_x, join_snp(n, ref_x, in_data, average_repeats=average_repeats)


def nf2frd(
    filepath: Union[str, PathLike],
    data: xr.DataArray,
    *args,
    compress: bool = False,
    version: Union[float, int] = 1,
    **kwargs,
) -> Path:
    """

    Parameters
    ----------
    filepath : str or Path
    data: xr.DataArray

    """
    data.attrs["type"] = "nearfield"
    data.attrs["version"] = version
    return xarray2frd(filepath, data, compress=compress)


def ps2frd(
    filepath: Union[str, PathLike],
    data: xr.DataArray,
    *args,
    compress: bool = False,
    version: Union[float, int] = 1,
    **kwargs,
) -> Path:
    """

    Parameters
    ----------
    filepath : str or Path
    data: xr.DataArray

    """
    data.attrs["type"] = "power-sweep"
    data.attrs["version"] = version
    return xarray2frd(filepath, data, compress=compress)


def read_htf(filepath: Union[str, PathLike]) -> xr.DataArray:
    """Read a Holzworth Trace File (phase noise analyzer)

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    DataArray

    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    if ext == ".htf":
        import xml.etree.ElementTree as etree

        file_ = open(filepath, "r", encoding="utf-8-sig")
        root = etree.fromstring(file_.read())

        traces = []
        for trace in root.findall("TRACE"):
            freq = None
            real = None
            imag = None
            band = None
            attrs = {}
            for element in trace:
                k = element.tag
                v = element.text
                if "_ARRAY" in k:
                    v = np.array([float(x) for x in v.split(",")])

                # Handle the important stuff
                if k == "FREQ_ARRAY":
                    freq = v
                elif k == "REAL_ARRAY":
                    real = v
                elif k == "IMAG_ARRAY":
                    imag = v
                elif k == "BAND_ARRAY":
                    band = v
                    _ = band  # not implemented....
                # TODO: Parse more fields
                else:
                    # Everything else is an attribute
                    try:
                        attrs[k] = float(v)
                    except ValueError:
                        attrs[k] = v

            traces.append(xr.DataArray(real + 1j * imag, coords=(("frequency", freq),), attrs=attrs))
    else:
        raise NotImplementedError(f"File type {ext} is not yet supported.")
    return traces


def read_meta_data(*fpaths: Union[str, PathLike], squeeze=True, recursive=False) -> dict:
    """Read the frf_meta_data from one or more touchstone (sNp) files.

    Parameters
    ----------
    fpaths : strings
        Paths to touchstone files. If multiple paths are provided they can
        either be provided individually or they must be unpacked from the
        containing list.
    squeeze : bool, opt
        If False, when a single path is provided the resulting array will have
        the shape (1, dict). If True, the unitary
        dimension will be removed. Defaults to True.
    recursive : bool, opt
        If True, recursively searches fpath for S2P files and passes list of
        file paths to be read.

    Returns
    -------
    d : array_like
        Dictionary (or array of dictionaries) from touchstone file.

    References
    ----------
    [1] https://www.microwaves101.com/images/downloads/touchstone_ver2_0.pdf

    """

    tempdir = None
    temppaths = fpaths
    fpaths = []
    md_list = []

    try:
        if recursive is True:
            x = []
            for fp in temppaths:
                fp = Path(fp)
                x.extend(fp.glob("**/*.s*p"))
            temppaths = x

        # Ensure all paths are of type Path
        temppaths = [Path(f) for f in temppaths]

        # Sort through the list of files and extract any zip files to tempdirs
        for fpath in temppaths:
            if fpath.suffix.lower() == ".zip":
                if tempdir is None:
                    tempdir = tempfile.TemporaryDirectory()
                extractdir = Path(tempdir.name) / fpath.name
                with zipfile.ZipFile(fpath, "r") as zip_ref:
                    zip_ref.extractall(str(extractdir))
                fpaths.extend(list(extractdir.glob("**/*.s*p")))
            else:
                fpaths.append(fpath)

        # Make sure all extensions match
        if len(set([f.suffix for f in fpaths])) > 1:
            raise ValueError("Touchstone files must all have same number of ports to be read")

        for fpath in fpaths:
            start = False
            stop = False
            comments = []
            with open(fpath, "r") as fobj:
                lines = fobj.readlines()

            for line in lines:
                line = line.rstrip()
                if start and stop:
                    break
                elif start and not stop:
                    if "! frf_meta_data" in line:
                        stop = True
                    else:
                        comments.append(line.strip("! "))
                elif not start and not stop:
                    if "! frf_meta_data" in line:
                        start = True
                    else:
                        pass
                else:
                    break

            if len(comments) == 0:
                md_list.append({})
            else:
                md_list.append(json.loads("\n".join(comments)))

        if squeeze and len(fpaths) == 1:
            md_list = md_list[0]
    except Exception as e:
        raise e from None
    finally:
        if tempdir is not None:
            tempdir.cleanup()

    return md_list


def read_nf(
    filename: Union[str, PathLike], xarray: bool = False
) -> Tuple[np.ndarray, Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """Import noise figure data from an agilent spectrum analyzer.

    Parameters
    ----------
    filename : str
    xarray : bool
        If true, nf and gain will be returned as xarrays
    Returns
    -------
    frequency : array_like
        Frequency [Hz]
    nf : array_like, DataArray
        Noise figure [dB]
    gain : array_like, DataArray
        Noise gain [dB]

    """
    filename = Path(filename)
    # Noise figure data has the following columns: RF freq, IF freq, NF, nf, Gain, Teff, Phot, Pcold
    data = np.genfromtxt(filename, delimiter=",", skip_header=41, comments="U")
    # The first half of the data is the corrected data
    # There is a middle, NaN line that can be tossed
    # The second half of the data is the uncorrected data
    num_pts = int((data.shape[0] - 1) / 2)

    # We're only interested in the corrected data
    data = data[:num_pts, :]
    freq = data[:, 0]
    nf = data[:, 2]
    gain = data[:, 4]

    if xarray:
        nf = xr.DataArray(nf, coords={"frequency": freq}, dims=("frequency"), attrs={"filepath": filename})
        gain = xr.DataArray(gain, coords={"frequency": freq}, dims=("frequency"), attrs={"filepath": filename})

    return freq, nf, gain


def read_touchstone(
    *fpaths: Union[str, PathLike],
    squeeze=True,
    recursive=False,
    xarray=False,
    interpolate=False,
) -> Tuple[np.ndarray, Union[np.ndarray, xr.DataArray]]:
    """Read one or more touchstone (sNp) files, or zipped sNp files.

    The s-parameter and x vectors will be grouped into a single vector.
    Files within zip archives may appear at any directory level and all
    s*p files will be opened.

    Parameters
    ----------
    fpaths : strings
        Paths to touchstone files. If multiple paths are provided they can
        either be provided individually or they must be unpacked from the
        containing list.
    squeeze : bool
        If False, when a single path is provided the resulting arrays will have
        the shape (1, len(freq)) and (1, len(freq), N, N). If True, the unitary
        dimension will be removed. Defaults to True.
    recursive : bool
        If True, recursively searches fpath for S2P files and passes list of
        file paths to be read.
    xarray : bool
        If True, `s` will be returned as an xarray DataArray.
    interpolate : bool
        If True, read_touchstone will interpolate and join the files on common
        overlapping frequency spans. Useful for mismatched files.

    Returns
    -------
    x : array_like
        Independent variable for touchstone measurement. This can be either
        frequency or output power depending upon the analyzer configuration.
        Shape: (len(fpaths), len(x)) or (len(x), )
    s : array_like
        Shape: (len(fpaths), len(x), N, N) or (len(x), N, N)

    References
    ----------
    [1] https://www.microwaves101.com/images/downloads/touchstone_ver2_0.pdf

    """

    tempdir = None
    temppaths = fpaths
    fpaths = []
    s_list = []
    x_list = []

    try:
        if recursive is True:
            x = []
            for fp in temppaths:
                fp = Path(fp)
                x.extend(fp.glob("**/*.s*p"))
            temppaths = x

        # Ensure all paths are of type Path
        temppaths = [Path(f) for f in temppaths]

        # Sort through the list of files and extract any zip files to tempdirs
        for fpath in temppaths:
            if fpath.suffix.lower() == ".zip":
                if tempdir is None:
                    tempdir = tempfile.TemporaryDirectory()
                extractdir = Path(tempdir.name) / fpath.name
                with zipfile.ZipFile(fpath, "r") as zip_ref:
                    zip_ref.extractall(str(extractdir))
                fpaths.extend(list(extractdir.glob("**/*.s*p")))
            else:
                fpaths.append(fpath)

        # Make sure all extensions match
        if len(set([f.suffix for f in fpaths])) > 1:
            raise ValueError("Touchstone files must all have same number of ports to be read")

        for fpath in fpaths:
            # Determine how many ports (n) are in the file.
            # Regex is required to support multi-digit port numbers (e.g. 10, 100)
            ext = fpath.suffix
            n = int(re.findall(r"\d+", ext)[0])

            # Read the file
            data = []  # List of data lines in touchstone file (strings)
            comments = []  # List of comment lines (strings)
            format_options = None  # Format option string
            smatrix = []  # Various s-parameter values
            x = []  # X values for the s-parameters
            count = n**2

            with open(fpath, "r") as fobj:
                lines = fobj.readlines()

            # Sort all the lines in the touchstone file. This is necessary because:
            #   A. Comment strings can appear throughout the entire file!
            #   B. I was lazy... for now.
            format_options = None
            for iline, line in enumerate(lines):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    # comment = re.match(RE_COMMENT, line)
                    # if comment:
                    #     comments.append(comment.groups()[0])
                    if line[0] == "!":
                        c = line[1:].strip()
                        comments.append(c)
                    elif not format_options:
                        format_options = re.match(RE_FORMAT, line.lower())
                        if format_options:
                            freq_unit = format_options.group("freq_unit")
                            # TODO: param unused
                            # param = format_options.group("param")
                            data_format = format_options.group("format")
                            # TODO: impedance currently ignored
                            # if format_options.group("z0") is None:
                            #     impedance = 50
                            # else:
                            #     impedance = format_options.group("z0")
                    # Kick out improperly escaped comment lines by looking for anything
                    # not part of a valid floating point value
                    elif re.match(RE_NOTDATA, line):
                        pass
                    else:
                        groups = [float(x.group()) for x in re.finditer(RE_FLOAT, line)]
                        values = [float(x) for x in groups]
                        data += values
                except Exception:
                    raise ValueError(f'Error parsing line {iline} of file {fpath}:\n"{line}"')

            data = np.array(data)
            # Expected shape of data
            if len(data) % (count * 2 + 1):
                raise ValueError(
                    f"Found unexpected number of data values. Expected {int(count * 2 + 1)} values per row."
                )
            shape = (int(len(data) / (count * 2 + 1)), int(count * 2 + 1))
            # Reshape data to shape like (401, 9) for s2p data
            # data[:,0] corresponds to the frequency vector
            # handles wrapped data
            data = data.reshape(shape)
            # Frequency
            x = data[:, 0] * SI_SCALAR[freq_unit]
            # Values
            c = data[:, 1:]
            c = c.flatten()
            if data_format == "ri":
                smatrix = c[0::2] + c[1::2] * 1j
            elif data_format == "db":
                smatrix = db2v(c[0::2]) * np.exp(1j * np.deg2rad(c[1::2]))
            elif data_format == "ma":
                smatrix = c[0::2] * np.exp(1j * np.deg2rad(c[1::2]))

            # Reshape based on frequency points and ports
            smatrix = np.reshape(smatrix, [len(x), n, n])

            # 2 port touchstone files are in the order: <N11>, <N21>, <N12>, <N22>
            # Where N21 is before N12. 3-port networks and greater don't do this
            if n == 2:
                temp = smatrix[:, 1, 0].copy()
                smatrix[:, 1, 0] = smatrix[:, 0, 1]
                smatrix[:, 0, 1] = temp

            x_list.append(x)
            s_list.append(smatrix)

        # When interpolate is selected, files will be resampled where they overlap with one another
        if interpolate:
            # Find overlapping bounds
            xmin = np.max([np.min(x) for x in x_list])
            xmax = np.min([np.max(x) for x in x_list])
            # Use largest number points. This is arbitrary, but maybe less so than just picking a number at random
            npts = np.max([len(x) for x in x_list])
            f = np.linspace(xmin, xmax, npts)
            x, s = _interp_snp(f, *zip([x for x in x_list], [s for s in s_list]))
        else:
            s = np.stack(s_list, axis=0)

        # Convert data into an xarray dataset
        # We don't want Path objects to end up in the DataArray attrs since these cannot be converted to .frd files
        # savemat will reject them!
        if xarray:
            s = xr.DataArray(
                s,
                coords={
                    "files": list(range(len(fpaths))),
                    "frequency": x,
                    "m": list(range(1, s.shape[-2] + 1)),
                    "n": list(range(1, s.shape[-1] + 1)),
                },
                dims=("files", "frequency", "m", "n"),
                attrs={"filepaths": [str(f) for f in fpaths]},
            )

        if squeeze and len(fpaths) == 1:
            s = s[0, ...]
    except Exception as e:
        raise e from None
    finally:
        if tempdir is not None:
            tempdir.cleanup()

    return x, s


def s2frd(
    filepath: Union[str, PathLike],
    s: Union[xr.DataArray, np.ndarray],
    *args,
    compress: bool = False,
    version: Union[float, int] = 1,
    **kwargs,
) -> Path:
    """Export multi-dimensional data to an FRD file.

    Parameters
    ----------
    filepath: str or Path
    s: DataArray or ndarray
        If DataArray, will be passed to xarray2frd
        If ndarray, args and kwargs will be used to construct a DataArray (see s2xarray)
    compress: bool
        Apply bzip2 compression if True. Resulting output suffix will be *.frd.bz2.
    """
    if not isinstance(s, xr.DataArray):
        s = s2xarray(s, *args, **kwargs)

    s.attrs["type"] = "s-parameter"
    s.attrs["version"] = version

    return xarray2frd(filepath, s, compress=compress)


# TODO: Are date, document, and version used in this function?
def s2xarray(
    s: np.ndarray,
    frequency: np.ndarray,
    temperature=20.0,
    power=np.nan,
    other_coords={},
    metadata={},
    date=None,
    document="",
    version=0.1,
) -> xr.DataArray:
    """Builds multi-dimensional DataArray for s-parameters.

    Parameters
    ----------
    s : np.ndarray
        S-parameters [complex voltage] with shape: (len(temperature), len(power), len(frequency), m, n)
        The temperature and power dimensions are optional, and only need to be present if temperature and power are
        non-scalar.
        If `other_coords` is provided, these dimensions must proceed `temperature`. As an example:
            other_coords = {'phase': ...}
            s.shape = (len(phase), len(temperature), len(power), len(frequency), m, n)
            Again, temperature and power dimensions must be present only if `temperature` and `power` are non-scalar.
    frequency: np.ndarray
        Frequency [Hz]
    temperature: np.ndarray
        Device temperature [C]
    power: np.ndarray
        Power [dBm]
        # TODO: Watts? Volts? I'm flexible here.
    other_coords: dict of np.ndarrays, optional
    metadata : dict
        Metadata. It's good for you.
    date : str, optional
        Date which should follow ISO-8601 format with UTC offset
    document : str
        String referencing FRF-XXXXX-ICD document number, default is "FRF-86272-ICD REV1"
    version : int, optional
        Defaults to the latest and greatest version. Allows exporting of older versions.

    """
    # Build up coordinates and dimensions
    # This has to match with the shape of s.

    coords = {
        "power": power,
        "temperature": temperature,
        "frequency": frequency,
        "m": np.arange(0, s.shape[-2]) + 1,
        "n": np.arange(0, s.shape[-1]) + 1,
    }
    coords = {**other_coords, **coords}

    # dimensions are only things that actually have a length
    # find them, and assign them
    dims = []
    for k, v in coords.items():
        if np.asarray(v).ndim != 0:
            dims.append(k)

    metadata["version"] = version
    metadata["type"] = "s-parameter"

    da = xr.DataArray(s, coords=coords, dims=dims, attrs=metadata)

    return da


def ts2frd(filepath, da, *args, compress: bool = False, version: Union[float, int] = 1, **kwargs) -> Path:
    da.attrs["type"] = "timeseries"
    da.attrs["version"] = version
    return xarray2frd(filepath, da, compress=compress)


def validate_frd(
    data: xr.DataArray,
    defaults: bool = True,
    required_coords: Optional[ArrayLike] = None,
    default_coords: dict = {},
    allowed_coords: Optional[dict] = None,
    required_attrs: Optional[ArrayLike] = None,
    default_attrs: dict = {},
    allowed_attrs: Optional[dict] = None,
    error: bool = False,
):
    """Check if data meets requirements of self specified frd file type.
    Reorders dimensions to meet requirements if necessary

    Parameters
    ----------
    data : DataArray
    defaults: bool, optional
        apply default values if missing
    required_coords: arraylike, optional
        Order of the list is applied to dimensions of data
        Required coords may be scalar (non-dimensioned)
    default_coords: dict, optional
        default values for coordinates.
    allowed_coords: dict, optional
        enumeration of all valid values for
    error: bool, optional
        elevate warnings to error

    Returns
    ----------
    DataArray

    """

    # TODO: add a flag for adding a dimension for every required coordinate (regardless of length)

    data = data.copy()

    if error:
        # save warning filters and add filter to elevate warnings starting with "ERROR:" to errors
        warning_filters = warnings.filters
        warnings.filterwarnings("error", message="ERROR:*")

    if "type" not in data.attrs:
        data.attrs["type"] = ""
        # Empty type indicates generic frb
        warnings.warn("attr 'type' missing")

    if "version" not in data.attrs:
        data.attrs["version"] = None
        # This will be overwritten with the latest version of the matching file type
        warnings.warn("attr 'version' missing")

    if data.attrs["type"] != "" and (
        required_coords or default_coords or allowed_coords or required_attrs or default_attrs or allowed_attrs
    ):
        # non-generic type generally shouldn't have custom requirements specified
        warnings.warn(
            f"Applying custom required/allowed/default_coords/attrs to non-generic data type '{data.attrs['type']}'."
            "Standard values will be ignored."
        )
    else:
        # use standard requirements based on file type

        if data.attrs["type"] == "":
            # https://confluence.firstrf.com/display/GI/Data+Format+Definition
            # generic type has no special validation
            data.attrs["version"] = 0

        elif data.attrs["type"] == "s-parameter":
            # https://confluence.firstrf.com/display/GI/S-Parameter+Data
            if data.attrs["version"] in (0.1, 1) or data.attrs["version"] is None:
                data.attrs["version"] = 1
                # Confluence says power then temperature but I think I saw the other way somethere too
                required_coords = ("power", "temperature", "frequency", "m", "n")
                default_coords = {"power": np.nan, "temperature": 20.0}
                required_attrs = ("z0",)
                default_attrs = {"z0": 50.0}
            else:
                warnings.warn(f"ERROR: Invalid 's-parameter' file format version: {data.attrs['version']}")

        elif data.attrs["type"] == "timeseries":
            # https://confluence.firstrf.com/display/GI/Skyler+Time+Series+Data+Format
            if data.attrs["version"] == 1 or data.attrs["version"] is None:
                data.attrs["version"] = 1
                required_coords = ("channel", "capture", "time", "time_offset")
                default_coords = {"channel": 0, "capture": 0, "time_offset": 0.0}
            else:
                warnings.warn(f"ERROR: Invalid 'timeseries' file format version: {data.attrs['version']}")

        elif data.attrs["type"] == "nearfield":
            # https://confluence.firstrf.com/display/GI/Nearfield+Data
            if data.attrs["version"] == 1:
                # TODO: verify nearfield coordinates. Confluence is contradictory
                # required_coords = ("polarization", "frequency", "m", "n")
                required_coords = ("polarization", "frequency", "y", "x")
                allowed_coords = {"polarization": ("x", "y", "lhcp", "rhcp")}
                if not {"x", "y", "lhcp", "rhcp"}.issuperset(data.coords["polarization"].data):
                    warnings.warn("ERROR: Polarization may only consist of 'x', 'y', 'lhcp', 'rhcp'")
            else:
                warnings.warn(f"ERROR: Invalid 'nearfield' file format version: {data.attrs['version']}")

        elif data.attrs["type"] == "pattern":
            # https://confluence.firstrf.com/display/GI/Pattern+%28Far-field%29+Data
            if data.attrs["version"] < 4:
                warnings.warn(f"ERROR: pattern format version {data.attrs['version']} not implemented")
                # # this jones_matrix stuff got copied from
                # # https://svn.firstrf.com/!/#Python/view/head/nsi/trunk/nsi/utils.py
                # if "jones_matrix" in data.dims:
                #     data.dims[data.dims.index("jones_matrix")] = "polarization"
                # if "jones_matrix" in d["coord_keys"]:
                #     del data.coords["jones_matrix"]
                #     data.coords["polarization"] = [str(x) for x in data.attrs["jones_vector_names"]]
            elif data.attrs["version"] == 4 or data.attrs["version"] is None:
                required_coords = ["polarization", "frequency"]
                required_coords.extend(
                    {
                        "phitheta": ("phi", "theta"),
                        "thetaphi": ("theta", "phi"),
                        "azel": ("az", "el"),
                        "elaz": ("el", "az"),
                        "uv": ("u", "v"),
                        "uvw": ("u", "v", "w"),
                        "kspace": ("kx", "ky"),
                        "trueview": ("tv_x", "tv_y"),
                    }[data.attrs["coordinate_frame"].lower()]
                )
            else:
                warnings.warn(f"ERROR: Invalid 'pattern' file format version: {data.attrs['version']}")

        elif data.attrs["type"] == "power-sweep":
            if data.attrs["version"] == 1:
                required_coords = ("pin", "frequency", "temperature", "time")
                default_coords = {"pin": np.nan, "temperature": 20.0}
                required_attrs = ("z0",)
                default_attrs = {"z0": 50.0}
            else:
                warnings.warn(f"ERROR: Invalid 'power-sweep' file format version: {data.attrs['version']}")

        else:
            warnings.warn(f"ERROR: Unknown FRD file type: '{data.attrs['type']}'")

    # type and version should both have been set by the above functions
    if data.attrs["type"] is None or data.attrs["version"] is None:
        raise ValueError("attrs 'type' and 'version' cannot be None")

    # clear default values if we're not supposed to use them
    if not defaults:
        default_coords = {}
        default_attrs = {}

    # ensure all dimensions have coordinates
    for dim in data.dims:
        if dim not in data.coords:
            warnings.warn(f"ERROR: Dimension '{dim}' has no coordinates")

    # verify existance of required coordinates
    if required_coords:
        for coord in required_coords:
            if coord not in data.coords:
                if coord in default_coords:
                    data.coords[coord] = default_coords[coord]
                    warnings.warn(f"Using default value '{default_coords[coord]}' for coordinate '{coord}'")
                else:
                    warnings.warn(f"ERROR: Required coordinate '{coord}' missing and has no default")

    # verify existance of required attributes
    if required_attrs:
        for attr in required_attrs:
            if attr not in data.attrs:
                if attr in default_attrs:
                    data.attrs[attr] = default_attrs[attr]
                    warnings.warn(f"Using default value '{default_attrs[attr]}' for attribute '{attr}'")
                else:
                    warnings.warn(f"ERROR: Required attribute '{attr}' missing and has no default")

    if required_coords:
        # reorder dimensions to put required coords last (if they're dimensioned)
        transpose_axes = (
            *[dim for dim in data.dims if dim not in required_coords],
            *[coord for coord in required_coords if coord in data.dims],
        )
        data = data.transpose(*transpose_axes)

    if error:
        # restore warning filters
        warnings.filters = warning_filters

    return data


def write_touchstone(
    fpath: Union[str, PathLike], freq: np.ndarray, smatrix: np.ndarray, comment=None, freq_unit="hz", format="ri"
):
    """Write a touchstone file (sNp where N>0) given frequency and
    s-parameter data.

    Based on the touchstone documentation found at:
    www.vhdl.org/ibis/touchstone_ver2.0/touchstone_ver2_0.pdf

    Parameters
    ----------
    fpath : str, Path
        Path to output file
    freq : array_like
        Frequency [Hz]
    smatrix : array_like
        M x M x N array of s-parameter values in real/imag format
    comment : list, dict
        Comments to include in header. If dict, it is converted to json
        to be placed in comments as frf_meta_data.
    freq_unit : {'hz', 'khz', 'mhz', 'ghz'}
        Desired frequency unit.
    format : {'ri', }
        `smatrix` format.

    References
    ----------
    [1] https://ibis.org/connector/touchstone_spec11.pdf

    """
    fpath = Path(fpath)
    with open(fpath, "w") as f:
        x = freq

        freq_unit = freq_unit.lower()
        if freq_unit not in ["hz", "khz", "mhz", "ghz"]:
            raise ValueError("Invalid frequency unit '{}'".format(freq_unit))

        if comment:
            if isinstance(comment, list):
                comment.insert(0, str(datetime.datetime.now()))
                for line_ in comment:
                    f.write("! " + str(line_).strip().replace("\n", "") + "\n")
            elif isinstance(comment, dict):
                f.write("! frf_meta_data\n")
                for line_ in json.dumps(comment, sort_keys=True, indent=4, separators=(",", ": ")).split("\n"):
                    f.write("! " + line_ + "\n")
                f.write("! frf_meta_data\n")
            else:
                pass
        else:
            f.write("! Non-documented touchstone file.  Generated " + str(datetime.datetime.now()) + "\n\n")

        # Output Touchstone format header:
        #   Freq - MHz, Param - S parameters, Format - Real/Imag, Normalized - 50 Ohms
        f.write("# " + freq_unit.upper() + " S RI R 50.0000\n\n")

        m, n, num_pts = smatrix.shape

        # crazy fix -> change this
        # TODO: Update this to handle shapes of any order. For instance: 2x2x100 or 100x2x2.
        if m > num_pts:
            temp = m
            m = num_pts
            num_pts = temp

        # maximum of four port measurements per line
        # loop through all frequencies and print associated s-parameters
        for ii, x in enumerate(x):
            f.write(str(x))
            # loop through s-parameter matrix
            for kk in range(n):
                for jj in range(m):
                    # For some inexplicable reason two port touchstone is ordered S11, S21, S12, S22 whereas
                    # touchstone with more than two ports are ordered S11, S12, S13, S14, ... etc. Thus, the following:
                    if m == 2:
                        real = smatrix[ii, jj, kk].real
                        imag = smatrix[ii, jj, kk].imag
                    else:
                        real = smatrix[ii, kk, jj].real
                        imag = smatrix[ii, kk, jj].imag
                    f.write("\t" + str(real) + "\t" + str(imag))
                    if ((kk + 1) % 4 == 0) and (m > 4):
                        f.write("\n")
            f.write("\n")


def xarray2frd(filepath: Union[str, PathLike], da: xr.DataArray, compress: bool = False, noneval=None) -> Path:
    """Export x-array to *.frd file, a FRF formatted *.mat file.

    Parameters
    ----------
    filepath : str, Path
        Desired output path. Note the extension may be modified by this call.
    da : DataArray
        Your data here.
    compress : bool
        Apply bzip2 compression if True. Resulting output suffix will be *.frd.bz2.
    noneval : any
        Replace all Nones with this value.
        This may be necessary since MATLAB (and therefore .frd) has no concept of None.
        In general, just avoid putting None in your xarray but this can be used particularly for old, unstructured data.

    Returns
    ----------
    Path
        Path of output file. `filepath` may have been modified to adhere to
        *.frd or *.frd.bz2 if compression is applied.
    """
    filepath = Path(filepath)
    # Ensure proper extension. expect ".frd", ".frd.bz2"
    if str(filepath).lower().endswith(".frd.bz2"):
        if not compress:
            warnings.warn('Got extension ".frd.bz2" but compress==False, changed to ".frd"')
        # pop the bz2, will be re-added later
        filepath = filepath.parent / (filepath.stem)  # stem will only drop the .bz2
    if filepath.suffix != ".frd":
        warnings.warn(f'File extension "{filepath.suffix}" changed to ".frd"')
    # Force the extension on the file
    filepath = filepath.parent / (filepath.stem + ".frd")

    def xarray2dict(da: xr.DataArray) -> dict:
        # Required Metadata
        if "date" not in da.attrs:
            da.attrs["date"] = (
                datetime.datetime.utcnow()
                .replace(tzinfo=datetime.timezone.utc)
                .replace(microsecond=0)
                .astimezone()
                .isoformat()
            )
        if "type" not in da.attrs:
            # generic frds have empty string type
            da.attrs["type"] = ""
        if "version" not in da.attrs:
            # validate_frd() will set version to latest if None
            da.attrs["version"] = None
        if "document" not in da.attrs:
            da.attrs["document"] = ""

        da = validate_frd(da, defaults=True, error=True)

        common_attrs = ("document", "date", "type", "version")

        d = dict()
        d["values"] = da.values
        d["dimensions"] = np.array(list(da.dims), dtype="object")
        d["coord_keys"] = np.array(list(da.coords.keys()), dtype="object")

        for attr in common_attrs:
            d[attr] = da.attrs[attr]

        for k, v in da.coords.items():
            if k in d:
                raise ValueError(f"Coordinate name '{k}' conflicts with frd metadata field")
            if v.dims == (k,):  # len(np.shape(v)) > 1:
                # logical (normal) dimensioned coordinate
                d[k] = np.asarray(v)
            elif v.dims == ():
                # singleton
                d[k] = np.asarray(v)
            else:
                # physical (non-dimensioned) coordinate
                d[k] = np.empty(2, dtype=object)
                d[k][0] = np.array(v.dims)  # a list of dimensions this coord varies with
                d[k][1] = np.asarray(v)  # N-dimensional array

        # attrs have given us a lot of issues since they can be arbitrary data types (including xarrays)
        # nested to arbitrary depth. Matlab data types are wack so we have to mangle everything first.
        # we will limit attributes to the following:
        #   - xr.DataArray: dict (matlab cell-array) containing the same stuff as if it were a frd on its own
        #   - arraylike: matlab array
        #   - dict: matlab cell-array (via numpy structured array)
        #   - string: matlab array of characters (bleh)
        #   - float: float
        #   - int: int
        #   - bool: bool

        def clean_attr(val):
            if val is None:
                if noneval is None:
                    raise ValueError("Nonetype found. MATLAB (and therefore .frd) has no concept of None")
                else:
                    warnings.warn(f"Nonetype found. Replacing with {noneval}")
                    val = noneval

            if isinstance(val, xr.DataArray):
                return xarray2dict(val)
            elif isinstance(val, dict):
                for k in val.keys():
                    if not isinstance(k, str):
                        raise ValueError(
                            f"Illegal dictionary key {k} found. "
                            "Dictionary keys must be strings for conversion to numpy structured arrays"
                        )
                return {k: clean_attr(v) for k, v in val.items()}
            elif isinstance(val, str) or isinstance(val, bytes):
                return np.asarray(val)
            else:
                try:
                    for ii in range(len(val)):
                        val[ii] = clean_attr(val[ii])
                    return val
                except TypeError:
                    return val

        for k, v in da.attrs.items():
            # Do not save the reserved attributes
            if k in RESERVED_ATTRS:
                continue
            elif k not in common_attrs:
                if k in d:
                    raise ValueError(f"Attribute name '{k}' conflicts")
                if not isinstance(k, str):
                    # TODO: restructure code so this check is not duplicated here and within clean_attr
                    raise ValueError(
                        f"Illegal dictionary key {k} found. "
                        "Dictionary keys must be strings for conversion to numpy structured arrays"
                    )
                d[k] = clean_attr(v)

        return d

    d = xarray2dict(da)

    # outfile and orignal filename may not match if compress==True because we need a working file.
    outfile = Path(filepath)

    # TODO: Save files in a temporary directory, then move them where they should go in the case of a compressed frd
    scipy.io.savemat(outfile, d, appendmat=False)

    # If compress, zip up the mat file and delete the original.
    # Scipy doesn't seem to work with IOStream objects so we have to actually touch the disk. Sad!
    if compress:
        # Open our brand new .mat and compress it
        with open(filepath, "rb") as file_:
            compressed = bz2.compress(file_.read())
        # Delete the .mat
        os.remove(filepath)
        # Add .bz2 which would have already been stripped if present in input filename
        outfile = filepath.parent / (str(filepath) + ".bz2")
        # Save it
        with open(outfile, "wb") as file_:
            file_.write(compressed)
    # Return the final output file path
    return outfile


def frd_hash(filepath: Union[str, PathLike]) -> str:
    """Get hash of frd file.

    Arguments
    ---------
    filepath:
        File to compute hash of.
    Returns:
        hash hexdigest
    """
    filepath = Path(filepath)
    if filepath.suffix.lower() in (".bz2",):
        file_ = bz2.open(filepath)
    else:
        file_ = open(filepath, mode="rb")
    # Compute hash
    # We read here to ensure hashes are based on the uncompressed file content
    source_hash = hashlib.md5(file_.read()).hexdigest()
    return source_hash


def read_rs_bin(filepath: Union[str, PathLike]) -> xr.DataArray:
    """Read Rohde & Schwarz RTO series oscilloscope binary data as xarray.

    Arguments
    ---------
    filepath:
        Source *.bin or *.Wfm.bin to read. Both are required to be in the same
        directory but either can be provided.

    Returns
    -------
        DataArray with dims matching the frd timeseries format with dims
        "capture", "channel", "time".
    """
    y, x, metadata = RTxReadBin(Path(filepath))
    da = xr.DataArray(
        y,
        dims=("time", "capture", "channel"),
        coords={
            "time": x,
            "capture": np.arange(y.shape[1]),
            "channel": np.arange(1, y.shape[2] + 1),
        },
    )
    da = da.transpose("capture", "channel", "time")
    da.attrs.update(metadata)
    # Attach source url in same manner as our FRD files
    da.attrs["_source_url"] = str(filepath)
    return da
