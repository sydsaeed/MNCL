import pytest

pytest.importorskip("torch_geometric")

from smoke_check import main


def test_full_mncl_smoke():
    main()
