from abc import ABC, abstractmethod
import glob
import sys
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/gcc_build')
import pyWaveformArchive as pwa

class BaseGatherDataUUSS(ABC):
    """ Abstract GatherData class that is inherited to specific cases of gathering data"""
    def __init__(self, archive_dir, processing_function):
        """
        Parameters
        ----------
        archive_dir: string
            directory that holds the archive hdf5 files
        """
        h5_archive_files = glob.glob(archive_dir + '/archive_????.h5')
        self.archive_manager = pwa.ArchiveManager()
        self.archive_manager.open_files_for_reading(h5_archive_files)
        self.processing_function = processing_function


    @abstractmethod
    def process_data(self, waveforms, pick_time, trace_cut_start = -3, trace_cut_end = 3):
        """ 
        Applies basic processing and interpolation of input signals.

        Parameters
        ----------
        waveform : waveform archive object
            The waveform to filter.
        pick_time : double
            The pick time in UTC seconds since the epoch.
        trace_cut_start : double
            The seconds before the pick time to begin the cut window.
        trace_cut_end : double
            The seconds after the pick time to end the cut window.

        Returns
        -------
        waveform : waveform archive object
            The filtered waveform.  Note, this can be none if an error
            was encountered or the input signal is too small.
        """
        pass

    @abstractmethod
    def create_waveform_df(self, catalog_file_name):
        """
        From the catalogs this creates a list of observed waveforms.

        Parameters
        ----------
        catalog : string
            Name of the waveform catalog. 

        Returns
        ----------
        catalog_df: Pandas DataFrame
            Observed waveforms
        """
        pass

    @abstractmethod
    def make_archive(self, catalog_df, output_file_root):
        """
        Creates an archive of time-series data (hdf5) with corresponding metadata (csv) for specified waveforms

        Parameters
        ----------
        catalog_df : DataFrame
            metadata for observed waveforms
        output_file_root: string
            path and name (not including file-type suffix) for the archive time-series and metadata files
        """
        pass


    def close(self):
        """ Close the archive manager used to access the archived time-series data"""
        self.archive_manager.close()
