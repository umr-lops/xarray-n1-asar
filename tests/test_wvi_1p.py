"""Tests for the ASA_WVI_1P reader against a synthetic product."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from n1_asar.asa_wvi_1p_reader import ASA_WVI_1P_Reader

import tests.conftest as builder

MJD_DATE = datetime(2000, 1, 1) + timedelta(
    days=builder.MJD[0], seconds=builder.MJD[1], microseconds=builder.MJD[2]
)


def test_mph_parsed(wvi_1p_reader):
    assert wvi_1p_reader.mph["PRODUCT"].strip() == "SYNTH_WVI_1P.N1"
    assert wvi_1p_reader.mph["SENSING_START"].startswith("2012-01-07")
    assert wvi_1p_reader.mph["ABS_ORBIT"] == "051539"
    assert int(wvi_1p_reader.mph["NUM_DSD"]) == 11


def test_sph_parsed(wvi_1p_reader):
    assert wvi_1p_reader.sph["SPH_DESCRIPTOR"].strip() == "SYNTH-SPH"
    assert wvi_1p_reader.sph["TX_RX_POLAR"] == "V/V"
    assert wvi_1p_reader.sph["NUM_WL_BINS"] == "0024"
    assert wvi_1p_reader.sph["FIRST_WL_BIN"].strip() == "1000.000000000"


def test_dsd_dataframe(wvi_1p_reader):
    dsd = wvi_1p_reader.dsd_df
    assert len(dsd) == 11
    names = list(dsd["DS_NAME"])
    assert any("SLC IMAGETTE MDS 000" in n for n in names)
    row = dsd[dsd["DS_NAME"].str.contains("CROSS SPECTRA MDS")].iloc[0]
    assert int(row["DS_OFFSET"][:-7]) == 50000
    assert int(row["DSR_SIZE"][:-7]) == 1061


def test_geolocation_ads(wvi_1p_reader):
    g = wvi_1p_reader.geolocation_ads_df
    assert len(g) == 4
    assert g["lat"].iloc[0] == 46.3
    assert g["lon"].iloc[0] == -1.2
    assert g["heading"].iloc[0] == 90.0
    assert g["date"].iloc[1] - g["date"].iloc[0] == np.timedelta64(1, "s")


def test_convert_mjd(wvi_1p_reader):
    dt = wvi_1p_reader.convert_mjd(builder.MJD)
    assert dt == MJD_DATE
    assert (dt.year, dt.month, dt.day) == (2150, 8, 2)
    assert dt.hour == 21
    assert wvi_1p_reader.convert_mjd((0, 0, 0)) == datetime(2000, 1, 1)


def test_read_line(wvi_1p_reader):
    fmt = ">iii b L %dh" % (builder.N_SAMPLES * 2)
    samples = [1 + (k % 399) for k in range(builder.N_SAMPLES * 2)]
    line = builder.struct.pack(fmt, 55000, 78631, 123456, 0, 42, *samples)
    date, qi, rln, sar = wvi_1p_reader.read_line(line, fmt)
    assert rln == 42
    assert qi == 0
    assert len(sar) == builder.N_SAMPLES
    assert np.iscomplexobj(sar)
    assert sar[0] == complex(samples[0], samples[1])


def test_read_imagette_mds(wvi_1p_reader):
    ds = wvi_1p_reader.read_imagette_mds(1)
    assert ds.sizes["line"] == 4
    assert ds.sizes["sample"] == builder.N_SAMPLES
    assert list(ds["line"].values) == [1, 2, 3, 4]
    assert float(ds["measurement_real"].values[0, 0]) == 1.0
    assert float(ds["measurement_imag"].values[0, 0]) == 2.0
    assert pd.Timestamp(ds["zero_doppler_time"].values[0]).hour == 21
    assert float(ds["quality_indicator"].max()) == 0.0


def test_read_cross_spectra_mds(wvi_1p_reader):
    ds = wvi_1p_reader.read_cross_spectra_mds(2)
    assert ds.sizes["direction"] == 18
    assert ds.sizes["wavelength"] == 24
    assert float(ds["spec_tot_energy"]) == 102.0
    assert float(ds["az_cutoff"]) == 12.0
    assert float(ds["max_real"]) == 100.0
    assert ds["real_spectra"].values[0, 0] == 100
    assert ds["real_spectra"].values[-1, -1] == 100
    assert ds["imag_spectra"].values[0, 0] == 50
    assert pd.Timestamp(ds["zero_doppler_time"].values[0]).second == 33


def test_read_sq_ads(wvi_1p_reader):
    ds = wvi_1p_reader.read_sq_ads(3)
    assert float(ds["thresh_chirp_broadening"]) == 4.0
    assert float(ds["thresh_chirp_sidelobe"]) == 5.0
    assert float(ds["lines_per_gap"]) == 10.0
    assert int(ds["num_gaps"]) == 23
    assert int(ds["num_missing_lines"]) == 33
    assert np.allclose(ds["input_mean"].values, [13.0, 14.0])
    assert np.allclose(ds["output_std_dev"].values, [43.0, 44.0])
    assert int(ds["tot_errors"]) == 103
    assert int(ds["land_flag"]) == 1
    assert pd.Timestamp(ds["time"].values[0]).second == 34


def test_read_processing_parameters_ads(wvi_1p_reader):
    ds = wvi_1p_reader.read_processing_parameters_ads(1)
    main = ds["main_parameters"].to_dataset()
    assert str(main["work_order_id"].item()).strip() == "SYNTH-ORDER"
    assert main["swath_num"].values == "IS1"
    assert float(main["range_spacing"]) == 3.0
    assert float(main["azimuth_spacing"]) == 2.0
    assert int(main["num_samples_per_line"]) == builder.N_SAMPLES
    assert main["data_type"].values == "SWORD"
    assert float(main["filter_coef_range"]) == 0.0
    orbit = ds["orbit_state_vectors"].to_dataset()
    assert orbit.sizes["state_vect_time"] == 5
    assert float(orbit["x_pos"].values[0]) == pytest.approx(63710.0)
    assert float(orbit["x_vel"].values[1]) == pytest.approx(1.0)
    assert float(orbit["z_vel"].values[4]) == pytest.approx(3.0)
    cal = ds["cal_info"].to_dataset()
    assert cal.sizes["dim_0"] == 32
    assert float(cal["avg_val_1a"].values[0]) == 1.0
    assert float(cal["avg_val_1a"].values[31]) == 32.0
    tie = ds["mid_line_tie_points"].to_dataset()
    assert float(tie["lats"].values[0]) == pytest.approx(46.000100)
    assert int(tie["mid_range_line_nums"]) == 100
    band = ds["bandwidth"].to_dataset()
    assert float(band["look_bw_range"]) == 100.0


def test_get_geolocation_ads_from_idx(wvi_1p_reader):
    ds = wvi_1p_reader.get_geolocation_ads_from_idx(2)
    assert float(ds["center_lat"]) == 46.32
    assert float(ds["center_long"]) == -1.18
    assert float(ds["heading"]) == 92.0
    assert int(ds["attach_flag"]) == 0


def test_get_datatree_structure(wvi_1p_reader):
    dt = wvi_1p_reader.get_datatree(0)
    for node in (
        "imagette",
        "cross_spectra",
        "geolocation_ads",
        "sq_ads",
        "processing_parameters",
    ):
        assert node in dt.children
    assert dt.attrs["platform"] == "ENVISAT"
    assert dt.attrs["product"].strip() == "SYNTH_WVI_1P.N1"
    assert dt.attrs["num_wl_bins"] == "0024"
    assert dt.attrs["tx_rx_polar"] == "V/V"


def test_datatree_index_suffix(wvi_1p_file):
    r = ASA_WVI_1P_Reader(wvi_1p_file + ":WV_003")
    try:
        dt = r.datatree
        assert dt is not None
        assert "imagette" in dt.children
        assert dt.attrs["platform"] == "ENVISAT"
    finally:
        r.file.close()


def test_display_digital_numbers(wvi_1p_datatree, tmp_path):
    from n1_asar.figures_wvi_1p import display_digital_numbers

    out = tmp_path / "dn.png"
    display_digital_numbers(wvi_1p_datatree, savepath=str(out))
    assert out.stat().st_size > 0
