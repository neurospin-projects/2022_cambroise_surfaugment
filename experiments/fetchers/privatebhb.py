from multiblock_fetcher import fetch_multiblock_wrapper


DEFAULTS = {
    "multiblock": {
        "test_size": 0, "seed": 42,
        "stratify": ["age", "sex", "site"], "discretize": ["age"],
        "blocks": ["surface-lh", "surface-rh"], "qc": None,
        "remove_nans": True, "remove_outliers": False,
        "allow_missing_blocks": False,
    }
}

def make_all_fetchers(datasetdir):
    fetchers = {}

    fetchers["multiblock"] = fetch_multiblock_wrapper(
        datasetdir=datasetdir,
        defaults=DEFAULTS["multiblock"])
    return fetchers
