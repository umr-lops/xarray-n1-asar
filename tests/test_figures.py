"""Tests for the plotting helpers in figures_wvi_1p."""
import numpy as np
import xarray as xr


def _fake_cross_spectra():
    rng = np.random.default_rng(0)
    real = rng.integers(0, 256, size=(18, 24)).astype(np.uint8)
    imag = rng.integers(0, 256, size=(18, 24)).astype(np.uint8)
    return xr.Dataset(
        data_vars={
            "real_spectra": (["direction", "wavelength"], real),
            "imag_spectra": (["direction", "wavelength"], imag),
            "min_real": ([], np.float64(0.0)),
            "max_real": ([], np.float64(1.0)),
            "min_imag": ([], np.float64(-1.0)),
            "max_imag": ([], np.float64(1.0)),
            "az_cutoff": ([], np.float64(12.0)),
        },
        coords={
            "direction": np.arange(18),
            "wavelength": np.geomspace(1000, 100, 24),
        },
    )


def test_get_polar_spectra_scaling():
    from n1_asar.figures_wvi_1p import get_polar_spectra

    ds = _fake_cross_spectra()
    ds["min_real"] = 10.0
    ds["max_real"] = 30.0
    ds["min_imag"] = -5.0
    ds["max_imag"] = 5.0
    spectra = get_polar_spectra(ds)
    assert np.iscomplexobj(spectra)
    expected_real = ds["real_spectra"].T.values / 255 * 20 + 10
    expected_imag = ds["imag_spectra"].T.values / 255 * 10 - 5
    assert np.allclose(spectra.real.values, expected_real)
    assert np.allclose(spectra.imag.values, expected_imag)


def test_symmetrize_polar_spectra():
    from n1_asar.figures_wvi_1p import get_polar_spectra, symmetrize_polar_spectra

    ds = _fake_cross_spectra()
    spectra = get_polar_spectra(ds)
    full = symmetrize_polar_spectra(spectra)
    assert full.sizes["direction"] == 2 * spectra.sizes["direction"]


def test_display_cross_spectra(wvi_1p_datatree, tmp_path):
    from n1_asar.figures_wvi_1p import display_cross_spectra

    out = tmp_path / "cross.png"
    display_cross_spectra(wvi_1p_datatree, savepath=str(out))
    assert out.stat().st_size > 0
