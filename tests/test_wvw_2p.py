"""Tests for the ASA_WVW_2P reader against a synthetic product."""
import pandas as pd


def test_mph_and_sph(wvw_2p_reader):
    assert wvw_2p_reader.mph["PRODUCT"].strip() == "SYNTH_WVW_2P.N1"
    assert wvw_2p_reader.sph["SPH_DESCRIPTOR"].strip() == "SYNTH-SPH"


def test_dsd_dataframe(wvw_2p_reader):
    dsd = wvw_2p_reader.dsd_df
    assert len(dsd) == 9
    assert any("OCEAN WAVE SPECTRA MDS" in n for n in list(dsd["DS_NAME"]))


def test_read_ocean_wave_spectra_mds(wvw_2p_reader):
    ds = wvw_2p_reader.read_ocean_wave_spectra_mds(1)
    assert "zero_doppler_time" in ds.coords
    assert float(ds["spec_tot_energy"]) == 101.0
    assert float(ds["spec_max_dir"]) == 45.0
    assert float(ds["SAR_wave_height"]) == 3.5
    assert float(ds["backscatter"]) == -20.0
    assert int(ds["confidence_swell"]) == 0
    assert int(ds["confidence_wind"]) == 1
    assert pd.Timestamp(ds["zero_doppler_time"].values[0]).second == 32


def test_read_sq_ads(wvw_2p_reader):
    ds = wvw_2p_reader.read_sq_ads(0)
    assert float(ds["thresh_chirp_sidelobe"]) == 2.0
    assert int(ds["num_missing_lines"]) == 30


def test_read_processing_parameters_ads(wvw_2p_reader):
    ds = wvw_2p_reader.read_processing_parameters_ads(0)
    main = ds["main_parameters"].to_dataset()
    assert main["swath_num"].values == "IS1"
    assert int(main["num_lines_proc"]) == 100


def test_get_datatree(wvw_2p_reader):
    dt = wvw_2p_reader.get_datatree(2)
    for node in (
        "ocean_wave_spectra",
        "geolocation_ads",
        "sq_ads",
        "processing_parameters",
    ):
        assert node in dt.children
    assert dt.attrs["platform"] == "ENVISAT"
    geo = dt["geolocation_ads"].to_dataset()
    assert float(geo["center_lat"]) == 46.32


def test_datatree_index_suffix(wvw_2p_file):
    from n1_asar.asa_wvw_2p_reader import ASA_WVW_2P_Reader

    r = ASA_WVW_2P_Reader(wvw_2p_file + ":WV_001")
    try:
        assert "ocean_wave_spectra" in r.datatree.children
    finally:
        r.file.close()
