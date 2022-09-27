"""
Script for testing
"""
from bg_rem import rm_tree
from pathlib import Path as p
from glob import glob

def test_rmdir():
    """
    Must erase dir
    """
    p.mkdir(p("test/test/test/test"), exist_ok=True, parents=True)
    with open("test/test/test.txt", "w") as f:
        f.write("test")
    rm_tree("test")
    assert len(glob("test/*")) == 0


test_rmdir()
