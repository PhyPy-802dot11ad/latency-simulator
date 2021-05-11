import os
import datetime
import pickle

import numpy as np
import pandas as pd


class Log():
    """Class representing a generic log.

    :param name: Log name.
    :param data: Logged data.
    :param col_names: Log column names.
    :param static_params: Any static data associated with the log.
    :param storage_dirpath: Path to where the log is saved.
    """

    def __init__(self, name, col_names, **kwargs):
        """Init new log.

        :param name: Log name.
        :param col_names: Log column names.
        :param **kwargs:
        """

        self.name = name
        self.data = np.empty((0, len(col_names)+1)) # +1 for timestamp
        self.col_names = col_names
        self.static_params = None
        self.set_storage_dirpath(**kwargs)

    def set_storage_dirpath(self, path='', subdir_name=datetime.datetime.now().isoformat()):
        """Set storage path and make empty subdir if it doesn't exist.

        :param path: Path to the parent log directory.
        :param subdir_name: Subdirectory name within the parent log directory.
        """

        self.storage_dirpath = os.path.join(path, subdir_name)
        # if not os.path.exists(self.storage_dirpath): os.mkdir(self.storage_dirpath)
        if not os.path.exists(self.storage_dirpath): os.makedirs(self.storage_dirpath)

    def append(self, *args):
        """Append a set of data to the log.

        :param args: Set of column entries to append.
        """

        row = [*args]
        self.append_row(row)

    def append_row(self, row):
        """Append a set of data to the log.

        :param args: Set of column entries to append.
        """

        self.data = np.concatenate( (self.data, [row]), axis=0 )

    def add_static_params(self, static_params):
        """Add static parameters to the log.

        :param static_params: Any static data associated with the log.
        """

        self.static_params = static_params

    def save_static_params(self):
        """Save (pickle) the static parameters. Skip if there isn't any static data.
        """

        if self.static_params is None: return
        filepath = os.path.join( self.storage_dirpath, f'{self.name}-static_params.pickle' )
        with open( filepath, 'wb' ) as f:
            pickle.dump(self.static_params, f)

    def save_log(self):
        """Save the data log to a npy file.
        """

        filepath = os.path.join( self.storage_dirpath, f'{self.name}-log.npy')
        np.save(filepath, self.data)

    def save_intermediate_to_csv(self):
        """Save the data log to a csv file.
        """

        df = pd.DataFrame(self.data[:,1:], index=self.data[:,0], columns=self.col_names) # Set timestamp as index
        # filepath = os.path.join(self.storage_dirpath, f'{self.name}-log.csv')
        filepath = os.path.join(self.storage_dirpath, f'{self.name}.csv')
        df.to_csv(filepath, index_label='Time (ns)')

    def save(self):
        """Save both the log (as csv) and it's staic data.
        """

        self.save_static_params()
        self.save_intermediate_to_csv()