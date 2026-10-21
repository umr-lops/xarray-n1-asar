"""Smoke tests: package imports and public reader classes are available."""

import n1_asar


def test_package_import():
    assert n1_asar.__name__ == "n1_asar"


def test_reader_classes_exist():
    from n1_asar.asa_wvi_1p_reader import ASA_WVI_1P_Reader
    from n1_asar.asa_wvw_2p_reader import ASA_WVW_2P_Reader

    assert ASA_WVI_1P_Reader is not None
    assert ASA_WVW_2P_Reader is not None
