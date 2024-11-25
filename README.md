xarray-n1-asar
================


Python library to read ASAR SLC product from ENVISAT mission (2002-2012).
Currently, the library supports the ASA_WVI_1P and ASA_WVW_2P file formats.

* Free software: MIT license


Installation
--------
```bash
pip install git+https://github.com/umr-lops/xarray-n1-asar.git
```

Usage
--------
In a python script:
```python
from n1_asar.asa_wvi_1p_reader import ASA_WVI_1P_Reader
from n1_asar.asa_wvw_2p_reader import ASA_WVW_2P_Reader

file_1p = 'ASA_WVI_1PNPDK20110108_145524_000007653098_00183_46318_5828.N1'
reader_1p = ASA_WVI_1P_Reader(file)

file_2p = 'ASA_WVW_2PPIFR20110102_001940_000001053098_00088_46223_8874.N1'
reader_2p = ASA_WVW_2P_Reader(file)
```






