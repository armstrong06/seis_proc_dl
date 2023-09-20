import obspy
from obspy.core.utcdatetime import UTCDateTime as UTC
import numpy as np

def match_channel_durations(st):
    """
    Make sure that all channels have the same start and end times, doesn't matter what they are 
    just make sure that they are the same and keep track of them. Could also pad/trim to exact hours?
    """
    # Go with the latest start time
    starttime = np.max([st[i].stats.starttime for i in range(3)])
    # Go with the earliest end time
    endtime = np.min([st[i].stats.endtime for i in range(3)])

    return st.trim(starttime, endtime), starttime, endtime 

def get_n_windows(npts, window_length, sliding_interval):
    return (npts-window_length)//sliding_interval + 1

def load_data(file, min_signal_percent=1, expected_file_duration_s=86400):
    """Reads in a miniseed file and check for gaps. Gaps are interpolated. 

    Args:
        file (str): Path to the miniseed file to read in.
        min_signal_percent (int, optional): Minimum percent signal required to process the file. Defaults to 1.
        max_duration_s (int, optional): Expected duration of the miniseed file in seconds. Defaults to 86400.

    Returns:
       tupple: (Obspy Stream, list of gaps)
    """ 

    # use obspy to load the miniseed file(s)
    st = obspy.read(file)

    sampling_rate = round(st[0].stats.sampling_rate)

    # If there is not enough signal in this day, skip the day
    total_npts = np.sum([st[i].stats.npts for i in range(len(st))])
    max_npts = expected_file_duration_s*round(sampling_rate)
    if (total_npts/max_npts)*100 < min_signal_percent:
        # Return the entire file period as a gap
        return None, format_edge_gaps(st, expected_file_duration_s, entire_file=True)

    # Save gaps so I know to ignore any detections in that region later
    gaps = st.get_gaps()
    if len(gaps) > 0:
        # Sometimes the gaps have different sampling rates and delta values, not totally sure why. 
        # The sampling rates do not equal npts/duration for the gap either.
        try:
            st.merge(fill_value='interpolate')
        except:
            delta = round(1/sampling_rate, 3)

            for tr in st:
                tr.stats.sampling_rate = sampling_rate
                tr.stats.delta = delta

            st.merge(fill_value='interpolate')

    # TODO: Do something with this meta info        
    starttime = st[0].stats.starttime
    endtime = st[0].stats.endtime
    npts = st[0].stats.npts

    # Check for gaps at the start/end of the day and save if they exist
    start_gap, end_gap = format_edge_gaps(st, expected_file_duration_s)
    if start_gap is not None:
        gaps.insert(0, start_gap)
    if end_gap is not None:
        gaps += [end_gap]

    return st, gaps 

    # TODO: Something to handle variations in the start/end time of the channels
    # if len(st) > 1:
    #     st, starttime, endtime = match_channel_durations(st)
    #     npts = 


def format_edge_gaps(st, max_duration_s, entire_file=False):
    """Checks for gaps greater than 1 second at the start and end of an Obspy Stream.
    Only looks at the first trace in the stream. *The desired endtime may be off by ~1 second*
    If a gap exists returns a list in the format of Obspy's get_gaps(), otherwise returns None.

    Args:
        st (Obspy Stream): stream to check for gaps
        max_duration_s (int): expected duration of the file in seconds
        entire_file (boolean, optional): True if the gap is the entire file. Defaults to False.

    Returns:
        tupple or list: tupple of gap info lists for the beginning and end of the trace or a list for the entire file gap. 
    """
    starttime = st[0].stats.starttime
    endtime = st[0].stats.endtime
    sampling_rate = st[0].stats.sampling_rate
    # TODO: this only works for days, not hours - fine for me
    # TODO: If the startime happens to be on the day before, it'll mess this up

    desired_start = UTC(starttime.year, starttime.month, starttime.day)
    desired_end = starttime + max_duration_s
    
    if entire_file:
        return [st[0].stats.network, st[0].stats.station, st[0].stats.location, st[0].stats.channel, 
                     desired_start, desired_end, max_duration_s, int(max_duration_s*sampling_rate)]

    # Compute duration of gaps at the start and end
    start_delta = starttime - desired_start
    end_delta = desired_end - endtime

    start_gap = None
    if start_delta > 1:
        start_gap = [st[0].stats.network, st[0].stats.station, st[0].stats.location, st[0].stats.channel, 
                     desired_start, starttime, start_delta, int(start_delta*sampling_rate)]

    end_gap = None
    if end_delta > 1:
        end_gap = [st[0].stats.network, st[0].stats.station, st[0].stats.location, st[0].stats.channel,
                    endtime, desired_end, end_delta, int(end_delta*sampling_rate)]

    return start_gap, end_gap

