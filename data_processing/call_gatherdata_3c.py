#sys.path.insert(0, "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis-proc-dl/detectors")
from gather_uuss_data import ThreeComponentGatherer
import sys
#sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/templateMatchingSource/rtseis/notchpeak4_gcc83_build/')
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/mlmodels/deepLearning/apply/np4_build')
import pyuussmlmodels as uuss
import os 

archive_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/archives/'
catalog_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/data/'
output_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/sPicker'
phase_type = "S"
waveform_halfwidth = 4
is_detector = False

if is_detector:
    processing_function = uuss.ThreeComponentPicker.ZRUNet.ProcessData()
else:
    processing_function = uuss.ThreeComponentPicker.ZCNN.ProcessData()

print(output_dir)
print("Phase type", phase_type)
print("Waveform halfwidth", waveform_halfwidth)
print("Detector?:", is_detector)
print(processing_function)

# Initialize 
gatherer = ThreeComponentGatherer(archive_dir, processing_function, is_detector)

infilename_roots = ['currentEarthquakeArrivalInformation','historicalEarthquakeArrivalInformation']
outfilename_roots = ['current_earthquake_catalog', 'historical_earthquake_catalog']
event_types =  ["le", "le"]

infilename_roots.append('currentBlastArrivalInformation')
outfilename_roots.append('current_blast_catalog')
event_types.append("qb")

for infilename_root, event_type, outfilename_root in zip(infilename_roots, event_types, outfilename_roots):
    print(f'loading {outfilename_root}...')
    filename = os.path.join(catalog_dir, f"{infilename_root}3C.csv")
    outfilename_root = os.path.join(output_dir, f"{phase_type}_{outfilename_root}_3c")
    gatherer.process_and_save_waveforms(filename, phase_type, outfilename_root, event_type, waveform_halfwidth)
    
gatherer.close()
