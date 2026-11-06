"""Test for tutorial functions."""

from __future__ import annotations

import numpy as np
import pytest

from lsapy import tutorial


@pytest.mark.network
class TestLoadDataset:
    def test_download_without_cache(self):
        ds = tutorial.load_dataset("drainage", cache=False)
        result = ds["drainage"].values[120, 20:30]
        expected = np.array([2, 3, 2, 2, 4, 4, 4, 4, 4, 3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_download_nz_regions(self):
        regions = tutorial.load_dataset("nz-regions")
        result = regions.iloc[0, :3].to_numpy()
        expected = np.array(["01", "Northland Region", "Northland Region"], dtype=object)
        np.testing.assert_array_equal(result, expected)

    def test_error_with_nz_regions_and_other_datasets(self):
        msg = "The 'nz-regions' dataset cannot be opened with other datasets. Please open it separately."
        with pytest.raises(ValueError, match=msg):
            tutorial.load_dataset(["drainage", "nz-regions"])
