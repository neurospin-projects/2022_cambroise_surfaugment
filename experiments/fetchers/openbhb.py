import os
import pandas as pd
from .multiblock_fetcher import fetch_multiblock_wrapper


DEFAULTS = {
    "multiblock": {
        "test_size": None, "seed": 42,
        "stratify": ["age", "sex", "site"], "discretize": ["age"],
        "blocks": ["surface-lh", "surface-rh"], "qc": None,
        "remove_nans": True, "remove_outliers": False,
        "allow_missing_blocks": False,
    }
}


def get_train_test_split_wrapper(datasetdir):
    """ Wrapper for function to get train and test subjects

        Parameters
        ----------
        files: dict, default FILES

        Returns
        -------
        get_train_test_split: function
            corresponding function

        """
    def get_train_test_split():
        """ Function to get train and test subjects

        Returns
        -------
        subjects: np.array
            train subjects
        subjects: np.array
            test subjects if available, None otherwise
        """
        subject_column_name = "participant_id"
        subjects = pd.read_table(os.path.join(
            datasetdir, "train_subjects.tsv"))[subject_column_name].values
        subjects_test = pd.read_table(os.path.join(
            datasetdir, "test_subjects.tsv"))[subject_column_name].values
        return subjects, subjects_test
    return get_train_test_split

def make_all_fetchers(datasetdir):
    fetchers = {}

    fetchers["multiblock"] = fetch_multiblock_wrapper(
        datasetdir=datasetdir,
        defaults=DEFAULTS["multiblock"],
        get_train_test_split=get_train_test_split_wrapper(datasetdir)
    )
    return fetchers