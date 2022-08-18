from gather_uuss_data import OneComponentGatherer
import sys
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/mlmodels/deepLearning/apply/np4_build')
import pyuussmlmodels as uuss
import os

archive_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/archives/'
catalog_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/data/'
output_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/fmPicker'
is_first_motion_data = True
waveform_halfwidth = 3

processing_function = uuss.OneComponentPicker.ZCNN.ProcessData()
# processing_function = uuss.ThreeComponentPicker.ZRUNet.ProcessData()
if is_first_motion_data:
    processing_function = uuss.FirstMotion.FMNet.ProcessData()

print("Using processing function", processing_function)
# Initialize 
gatherer = OneComponentGatherer(archive_dir, processing_function)
infilename_roots = ['currentEarthquakeArrivalInformation', 'currentBlastArrivalInformation', 'historicalEarthquakeArrivalInformation']
outfilename_roots = ['current_earthquake_catalog', 'current_blast_catalog', 'historical_earthquake_catalog']
for infilename_root, event_type, outfilename_root in zip(infilename_roots, ['le', 'qb', 'le'], outfilename_roots):
    print(f'loading {outfilename_root}...')
    filename_3c = os.path.join(catalog_dir, f'{infilename_root}3C.csv')
    filename_1c = os.path.join(catalog_dir, f'{infilename_root}1C.csv')
    outfilename_root = os.path.join(output_dir, f'{outfilename_root}_1c')
    drop_down = False
    if (event_type == "qb") and is_first_motion_data:
        drop_down = True

    gatherer.process_and_save_waveforms(filename_3c, filename_1c, outfilename_root, event_type,
                                        is_first_motion_data=is_first_motion_data, drop_down=drop_down,
                                        halfwindow_length=waveform_halfwidth)
    
gatherer.close()
