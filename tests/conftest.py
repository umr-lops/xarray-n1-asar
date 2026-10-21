"""Fixtures building synthetic ASA_WV product files.

The binary layout mirrors real ENVISAT ASAR products (MPH, SPH, DSD table,
MDS/ADS payloads) so the readers can be exercised hermetically in CI,
without depending on any external data store.

Field layouts were validated against real products:
- MPH: 1247 bytes, SPH: 901 bytes, DSD record: 280 bytes
- DSD offset/size: '+' + 20 digits + '<bytes>' (28 bytes)
- DSD dsr_size / MPH dsd_size: '+' + 10 digits + '<bytes>' (18 bytes)
- DSD num_dsr / MPH num_dsd: '+' + 10 digits (11 bytes)
- struct sizes: geo=25, sq=252, cross=1061, ocean wave=1061,
  processing params=3959, imagette line=2021
"""
from __future__ import annotations

import re
import struct
from typing import TYPE_CHECKING, Any, Iterator

import matplotlib
import pytest
import xarray as xr

if TYPE_CHECKING:
    from n1_asar.asa_wvi_1p_reader import ASA_WVI_1P_Reader
    from n1_asar.asa_wvw_2p_reader import ASA_WVW_2P_Reader

matplotlib.use("Agg")

MPH_SIZE = 1247
SPH_SIZE = 901
DSD_SIZE = 280
MJD = (55000, 78631, 123456)
NUM_IMAGETTES = 4
NUM_LINES = 4
IMAGETTE_DSR_SIZE = 2021
N_SAMPLES = (IMAGETTE_DSR_SIZE - 17) // 4  # 501 complex samples per line

GEO_FMT = ">iii ? l l f"
SQ_FMT = ">3i 12? 7x 15f L 15x 10f L 16x 6? 4x 4f L 2f 12x 5f 12x"
CROSS_FMT = ">3i ? 30f 64x 432B 432B"
OWS_FMT = ">3i ? 2f 4x 9f 56x 2f 8x 6f H 3f H 24x 432B 432B"
PP_FMT = (
    ">3i ? 3i 12c f 3c 3f 2L 5c L f 43x 15? 6x 4L 14f 4? 4f 92x 32x 2L 3i 20x 60h 60x"
    "10L 26x 10f 5L 40f 5h 5f 5L 62x L 3f h 7c f 10f 8f 128x 60x L h 2f 7c 6f 68x 2f 8x"
    "5f 5L 64x 12x 4f 16x f 48x 28c 64x 4L 4f 5L 3i 16x 3i 6l 3i 6l 3i 6l 3i 6l 3i 6l"
    "64x 7f c 13x 6f c f 7c 4x 352f 16x 3i 3L 6f 6l 3i L 3L 6f 6l 3i L 3L 6f 6l 12f H 3f 12x 33f 14x"
)

GEO_DSR_SIZE = struct.calcsize(GEO_FMT)
SQ_DSR_SIZE = struct.calcsize(SQ_FMT)
CROSS_DSR_SIZE = struct.calcsize(CROSS_FMT)
OWS_DSR_SIZE = struct.calcsize(OWS_FMT)
PP_DSR_SIZE = struct.calcsize(PP_FMT)

assert (GEO_DSR_SIZE, SQ_DSR_SIZE, CROSS_DSR_SIZE, OWS_DSR_SIZE, PP_DSR_SIZE) == (
    25,
    252,
    1061,
    1061,
    3959,
)


def _num20(v: int) -> bytes:
    """DSD offset/size field: 28 bytes."""
    return ("+" + str(v).zfill(20) + "<bytes>").encode()


def _num11(v: int) -> bytes:
    """DSD num_dsr / MPH num fields: 11 bytes."""
    return ("+" + str(v).zfill(10)).encode()


def _num18(v: int) -> bytes:
    """DSD dsr_size / MPH dsd_size fields: 18 bytes."""
    return ("+" + str(v).zfill(10) + "<bytes>").encode()


def _value_map(fmt: str) -> list[tuple[int, str]]:
    """Return a list of (byte_offset, typechar) for each unpacked value."""
    body = fmt[1:]  # drop '>'
    result: list[tuple[int, str]] = []
    off = 0
    i = 0
    while i < len(body):
        ch = body[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            m = re.match(r"\d+", body[i:])
            assert m is not None
            count = int(m.group())
            i += len(m.group())
            ch = body[i]
            i += 1
        else:
            count = 1
            i += 1
        if ch == "x":
            off += count
            continue
        size = struct.calcsize(">" + ch)
        for _ in range(count):
            result.append((off, ch))
            off += size
    return result


def _build_block(fmt: str, values: dict[int, Any]) -> bytes:
    """Build a binary block, setting fields by unpacked-value index.

    ``values`` maps an unpacked-value index to a Python value. For 'c' fields
    the value is an int in 0..255 (one byte). Unset fields stay zero.
    """
    vmap = _value_map(fmt)
    buf = bytearray(struct.calcsize(fmt))
    for k, value in values.items():
        off, ch = vmap[k]
        size = struct.calcsize(">" + ch)
        if ch == "c":
            value = bytes([value])
        buf[off : off + size] = struct.pack(">" + ch, value)
    return bytes(buf)


def _mjd_values(i: int) -> dict[int, float]:
    """MJD triple for row i, as a dict of unpacked indices -> numeric values."""
    return {0: MJD[0], 1: MJD[1] + i, 2: MJD[2]}


def _geo_block(num_rows: int) -> bytes:
    out = b""
    for i in range(num_rows):
        out += _build_block(
            GEO_FMT,
            {
                **_mjd_values(i),
                3: 0,  # attach_flag
                4: int((46.3 + 0.01 * i) * 1e6),  # lat (micro-deg)
                5: int((-1.2 + 0.01 * i) * 1e6),  # lon (micro-deg)
                6: 90.0 + i,  # heading
            },
        )
    return out


def _sq_block(num_rows: int) -> bytes:
    out = b""
    for i in range(num_rows):
        vals = _mjd_values(i)
        vals[3] = 0  # attach_flag
        vals[15] = 1.0 + i  # thresh_chirp_broadening
        vals[16] = 2.0 + i  # thresh_chirp_sidelobe
        vals[30] = 7 + i  # lines_per_gap
        vals[31] = 10.0 + i  # input_mean[0]
        vals[32] = 11.0 + i  # input_mean[1]
        vals[33] = 12.0 + i  # input_std_dev[0]
        vals[34] = 13.0 + i  # input_std_dev[1]
        vals[35] = 20 + i  # num_gaps
        vals[36] = 30 + i  # num_missing_lines / output_mean[0]
        vals[37] = 31 + i  # output_mean[1]
        vals[38] = 40 + i  # output_std_dev[0]
        vals[39] = 41 + i  # output_std_dev[1]
        vals[41] = 100 + i  # tot_errors
        vals[42] = 1  # land_flag
        vals[52] = 50 + i  # az_cutoff_iteration_thresh
        vals[55] = 60.0 + i  # look_conf
        out += _build_block(SQ_FMT, vals)
    return out


def _cross_block(num_rows: int) -> bytes:
    out = b""
    for i in range(num_rows):
        vals = _mjd_values(i)
        vals[3] = 0  # quality_flag
        vals[4] = 2.0  # range_spectral_res
        vals[5] = 3.0  # az_spectral_res
        vals[7] = 100.0 + i  # spec_tot_energy
        vals[8] = 50.0  # spec_max_energy
        vals[9] = 45.0  # spec_max_dir
        vals[10] = 10.0  # spec_max_wl
        vals[12] = 12.0  # az_cutoff
        vals[30] = -5.0  # min_imag
        vals[31] = 5.0  # max_imag
        vals[32] = 0.0  # min_real
        vals[33] = 100.0  # max_real
        for j in range(432):
            vals[34 + j] = 100  # real_spectra
            vals[466 + j] = 50  # imag_spectra
        out += _build_block(CROSS_FMT, vals)
    return out


def _ows_block(num_rows: int) -> bytes:
    out = b""
    for i in range(num_rows):
        vals = _mjd_values(i)
        vals[3] = 0  # quality_flag
        vals[4] = 2.0  # range_spectral_res
        vals[5] = 3.0  # az_spectral_res
        vals[6] = 100.0 + i  # spec_tot_energy
        vals[7] = 50.0  # spec_max_energy
        vals[8] = 45.0  # spec_max_dir
        vals[9] = 10.0  # spec_max_wl
        vals[11] = 12.0  # az_cutoff
        vals[15] = -1.0  # min_spectrum
        vals[16] = 10.0  # max_spectrum
        vals[17] = 8.0  # wind_speed
        vals[18] = 270.0  # wind_direction
        vals[20] = 3.5  # SAR_wave_height
        vals[22] = -20.0  # backscatter
        vals[23] = 0  # confidence_swell
        vals[27] = 1  # confidence_wind
        for j in range(432):
            vals[28 + j] = 100  # real_spectra
            vals[460 + j] = 50  # imag_spectra
        out += _build_block(OWS_FMT, vals)
    return out


def _pp_record(i: int) -> bytes:
    vals = _mjd_values(i)
    vals[4] = MJD[0]
    vals[5] = MJD[1] + i + 1
    vals[6] = MJD[2]
    for k, c in enumerate("SYNTH-ORDER "):  # work_order_id (12c)
        vals[7 + k] = ord(c)
    vals[19] = 0.5  # time_diff
    for k, c in enumerate("IS1"):  # swath_num (3c)
        vals[20 + k] = ord(c)
    vals[23] = 3.0  # range_spacing
    vals[24] = 2.0  # azimuth_spacing
    vals[25] = 0.001  # line_time_interval
    vals[26] = NUM_LINES  # num_output_lines
    vals[27] = N_SAMPLES  # num_samples_per_line
    for k, c in enumerate("SWORD"):  # data_type (5c)
        vals[28 + k] = ord(c)
    vals[222] = 500000.0  # range_ref
    vals[223] = 65000000.0  # range_samp_rate
    vals[224] = 54020000.0  # radar_freq
    vals[225] = 1  # num_looks_range
    for k, c in enumerate("HAMMING"):  # filter_range (7c)
        vals[226 + k] = ord(c)
    vals[234] = 100.0  # look_bw_range
    vals[252] = 100  # num_lines_proc
    vals[253] = 1  # num_look_az
    for k, c in enumerate("KAISER "):  # filter_az (7c)
        vals[256 + k] = ord(c)
    vals[264] = 1.0
    vals[265] = 2.0
    vals[266] = 3.0  # az_fm_rate
    vals[269] = 1.2  # proc_scaling_fact
    vals[270] = 0.8  # ext_scal_fact
    for k, c in enumerate("FBAQ"):  # echo_comp (4c)
        vals[286 + k] = ord(c)
    for k, c in enumerate("8/4"):  # echo_comp_ratio (3c)
        vals[290 + k] = ord(c)
    for k, c in enumerate("REF     "):  # norm_source (7c)
        vals[391 + k] = ord(c)
    # orbit state vectors: 5 blocks of (3i mjd, 6l [x,y,z pos, x,y,z vel])
    for s in range(5):
        base = 330 + s * 9
        vals[base] = MJD[0]
        vals[base + 1] = MJD[1] + s
        vals[base + 2] = MJD[2]
        vals[base + 3] = 6371000  # x_pos (*1e-2 -> 63710.0)
        vals[base + 4] = 1234567  # y_pos
        vals[base + 5] = 7654321  # z_pos
        vals[base + 6] = 100000  # x_vel (*1e-5 -> 1.0)
        vals[base + 7] = 200000  # y_vel
        vals[base + 8] = 300000  # z_vel
    # cal_info: 32 rows of 11 floats starting at index 398
    for r in range(32):
        vals[398 + r * 11 + 6] = 1.0 + r  # avg_val_1a
    # mid_line_tie_points
    vals[768] = MJD[0]
    vals[769] = MJD[1] + i
    vals[770] = MJD[2]
    vals[771] = 100  # mid_range_line_nums
    vals[781] = 46000100  # lats[0] (*1e-6 -> 46.0001)
    vals[782] = 46000200
    vals[783] = 46000300
    vals[784] = -1000100  # longs[0]
    vals[785] = -1000200
    vals[786] = -1000300
    return _build_block(PP_FMT, vals)


def _pp_block(num_rows: int) -> bytes:
    return b"".join(_pp_record(i) for i in range(num_rows))


def _imagette_block(num_lines: int, first_line: int = 1) -> bytes:
    out = b""
    for j in range(num_lines):
        vals = {0: MJD[0], 1: MJD[1] + j, 2: MJD[2], 3: 0, 4: first_line + j}
        for k in range(N_SAMPLES * 2):
            vals[5 + k] = ((j * 1000 + k) % 399) + 1
        out += _build_block(">iii b L %dh" % (N_SAMPLES * 2), vals)
    return out


def _mph(product: str, num_dsd: int, total_size: int) -> bytes:
    b = bytearray(b" " * MPH_SIZE)

    def put(start: int, text: str) -> None:
        b[start : start + len(text)] = text.encode()

    put(9, product.ljust(62))
    put(236, "2026-01-01 00:00:00.000000")
    put(351, "2012-01-07 21:57:31.000000")
    put(394, "2012-01-07 21:57:32.000000")
    put(84, "1")
    put(470, "A")
    put(478, "0100")
    put(493, "000232")
    put(510, "051539")
    put(536, "2012-01-07 21:57:31.000000")
    put(770, "SV")
    put(829, "2012-01-07 21:57:31.000000")
    put(874, "00123456789")
    put(897, "000000000100000")
    put(956, "2012-01-07 21:57:31.000000")
    put(995, "+000")
    put(1009, "0")
    put(1064, "0")
    b[1075:1103] = _num20(total_size)
    b[1113:1131] = _num18(SPH_SIZE)
    b[1140:1151] = _num11(num_dsd)
    b[1161:1179] = _num18(DSD_SIZE)
    b[1194:1205] = _num11(num_dsd)
    return bytes(b)


def _sph() -> bytes:
    b = bytearray(b" " * SPH_SIZE)

    def put(start: int, text: str) -> None:
        b[start : start + len(text)] = text.encode()

    put(16, "SYNTH-SPH")
    put(63, "2012-01-07 21:57:31.000000")
    put(108, "2012-01-07 21:57:32.000000")
    put(197, "001")
    put(211, "002")
    put(222, "Ascending")
    put(247, "V/V")
    put(265, "NONE")
    put(336, "0018")
    put(353, "0024")
    put(372, "0.000000")
    put(410, "10.000000")
    put(448, "1000.000000000")
    put(479, "100.0000000000")
    put(558, "0.000000")
    put(585, "0.000000")
    put(618, "0004")
    put(637, "0")
    put(652, "1")
    put(660, "1")
    put(672, "0")
    put(719, "0003")
    put(738, "00000000100")
    put(766, "00000000036")
    put(792, "0.000000")
    put(828, "0000")
    put(848, "0000")
    put(868, "0004")
    put(886, "0004")
    return bytes(b)


def _dsd(name: str, offset: int, size: int, num_dsr: int, dsr_size: int) -> bytes:
    b = bytearray(b" " * DSD_SIZE)
    b[9 : 9 + len(name)] = name.encode()
    b[47:48] = b"R"
    b[59 : 59 + len(name)] = name.encode()
    b[133:161] = _num20(offset)
    b[170:198] = _num20(size)
    b[207:218] = _num11(num_dsr)
    b[228:246] = _num18(dsr_size)
    return bytes(b)


def _assemble(mph: bytes, sph: bytes, dsds: list[bytes], payloads: dict[int, bytes], total_size: int) -> bytes:
    buf = bytearray(total_size)
    buf[0:MPH_SIZE] = mph
    buf[MPH_SIZE : MPH_SIZE + SPH_SIZE] = sph
    dsd_start = MPH_SIZE + SPH_SIZE
    dsd_table = b"".join(dsds)
    buf[dsd_start : dsd_start + len(dsd_table)] = dsd_table
    for offset, data in payloads.items():
        buf[offset : offset + len(data)] = data
    return bytes(buf)


def build_wvi_1p(num_imagettes: int = NUM_IMAGETTES, num_lines: int = NUM_LINES) -> bytes:
    sq_off, geo_off, pp_off, cross_off = 10000, 20000, 30000, 50000
    imag_offsets = [60000 + 10000 * i for i in range(num_imagettes)]
    payloads = {
        sq_off: _sq_block(num_imagettes),
        geo_off: _geo_block(num_imagettes),
        pp_off: _pp_block(num_imagettes),
        cross_off: _cross_block(num_imagettes),
    }
    for i, off in enumerate(imag_offsets):
        payloads[off] = _imagette_block(num_lines)
    dsds = [
        _dsd("SQ ADS", sq_off, num_imagettes * SQ_DSR_SIZE, num_imagettes, SQ_DSR_SIZE),
        _dsd(
            "GEOLOCATION ADS",
            geo_off,
            num_imagettes * GEO_DSR_SIZE,
            num_imagettes,
            GEO_DSR_SIZE,
        ),
        _dsd(
            "PROCESSING PARAMS ADS",
            pp_off,
            num_imagettes * PP_DSR_SIZE,
            num_imagettes,
            PP_DSR_SIZE,
        ),
        _dsd(
            "CROSS SPECTRA MDS",
            cross_off,
            num_imagettes * CROSS_DSR_SIZE,
            num_imagettes,
            CROSS_DSR_SIZE,
        ),
    ]
    for i, off in enumerate(imag_offsets):
        dsds.append(
            _dsd(
                "SLC IMAGETTE MDS %03d" % i,
                off,
                num_lines * IMAGETTE_DSR_SIZE,
                num_lines,
                IMAGETTE_DSR_SIZE,
            )
        )
    for _ in range(3):
        dsds.append(_dsd("LEVEL 0 PRODUCT", 0, 0, 0, 0))
    total = max(imag_offsets) + num_lines * IMAGETTE_DSR_SIZE + 1024
    mph = _mph("SYNTH_WVI_1P.N1", len(dsds), total)
    return _assemble(mph, _sph(), dsds, payloads, total)


def build_wvw_2p(num_imagettes: int = NUM_IMAGETTES) -> bytes:
    sq_off, geo_off, pp_off, ows_off = 10000, 20000, 30000, 50000
    payloads = {
        sq_off: _sq_block(num_imagettes),
        geo_off: _geo_block(num_imagettes),
        pp_off: _pp_block(num_imagettes),
        ows_off: _ows_block(num_imagettes),
    }
    dsds = [
        _dsd("SQ ADS", sq_off, num_imagettes * SQ_DSR_SIZE, num_imagettes, SQ_DSR_SIZE),
        _dsd(
            "GEOLOCATION ADS",
            geo_off,
            num_imagettes * GEO_DSR_SIZE,
            num_imagettes,
            GEO_DSR_SIZE,
        ),
        _dsd(
            "PROCESSING PARAMS ADS",
            pp_off,
            num_imagettes * PP_DSR_SIZE,
            num_imagettes,
            PP_DSR_SIZE,
        ),
        _dsd(
            "OCEAN WAVE SPECTRA MDS",
            ows_off,
            num_imagettes * OWS_DSR_SIZE,
            num_imagettes,
            OWS_DSR_SIZE,
        ),
    ]
    for _ in range(5):
        dsds.append(_dsd("LEVEL 0 PRODUCT", 0, 0, 0, 0))
    total = ows_off + num_imagettes * OWS_DSR_SIZE + 1024
    mph = _mph("SYNTH_WVW_2P.N1", len(dsds), total)
    return _assemble(mph, _sph(), dsds, payloads, total)


@pytest.fixture(scope="session")
def wvi_1p_file(tmp_path_factory: pytest.TempPathFactory) -> str:
    p = tmp_path_factory.mktemp("asar") / "SYNTH_WVI_1P.N1"
    p.write_bytes(build_wvi_1p())
    return str(p)


@pytest.fixture(scope="session")
def wvw_2p_file(tmp_path_factory: pytest.TempPathFactory) -> str:
    p = tmp_path_factory.mktemp("asar") / "SYNTH_WVW_2P.N1"
    p.write_bytes(build_wvw_2p())
    return str(p)


@pytest.fixture()
def wvi_1p_reader(wvi_1p_file: str) -> "Iterator[ASA_WVI_1P_Reader]":
    from n1_asar.asa_wvi_1p_reader import ASA_WVI_1P_Reader

    r = ASA_WVI_1P_Reader(wvi_1p_file)
    yield r
    r.file.close()


@pytest.fixture()
def wvw_2p_reader(wvw_2p_file: str) -> "Iterator[ASA_WVW_2P_Reader]":
    from n1_asar.asa_wvw_2p_reader import ASA_WVW_2P_Reader

    r = ASA_WVW_2P_Reader(wvw_2p_file)
    yield r
    r.file.close()


@pytest.fixture()
def wvi_1p_datatree(wvi_1p_reader: ASA_WVI_1P_Reader) -> xr.DataTree:
    return wvi_1p_reader.get_datatree(0)
