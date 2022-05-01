#sys.path.insert(0, "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis-proc-dl/detectors")
from gather_uuss_data import ThreeComponentGatherer
import sys
#sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/templateMatchingSource/rtseis/notchpeak4_gcc83_build/')
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/mlmodels/deepLearning/apply/np4_build')
import pyuussmlmodels as uuss
import os 

archive_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/archives/'
catalog_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/data/'
output_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/uuss2021'
phase_type = "P"
n_comps = 1
waveform_halfwidth = 10

processing_function = uuss.ThreeComponentPicker.ZRUNet.ProcessData()

# Initialize 
gatherer = ThreeComponentGatherer(archive_dir, processing_function)

infilename_roots = ['currentEarthquakeArrivalInformation','historicalEarthquakeArrivalInformation']
outfilename_roots = ['current_earthquake_catalog', 'historical_earthquake_catalog']
event_types = ["le", "le"]

if phase_type == "P":
    infilename_roots.append('currentBlastArrivalInformation')
    outfilename_roots.append('current_blast_catalog')
    event_types.append("qb")

for infilename_root, event_type, outfilename_root in zip(infilename_roots, event_types, outfilename_roots):
    print(f'loading {outfilename_root}...')
    filename = os.path.join(catalog_dir, f"{infilename_root}{n_comps}C.csv")
    outfilename_root = os.path.join(output_dir, f"{phase_type}_{outfilename_root}_{n_comps}c")
    gatherer.process_and_save_waveforms(filename, phase_type, outfilename_root, event_type, waveform_halfwidth)
    
gatherer.close()
