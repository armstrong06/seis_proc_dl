from data_processing.gatherdata import OneComponentGatherData
import pyuussmlmodels as uuss
import os 

archive_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/archives/'
catalog_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/data/'
output_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/firstMotion'
is_first_motion_data = True

processing_function = uuss.OneComponentPicker.ZCNN.ProcessData()
if is_first_motion_data:
    processing_function = uuss.FirstMotion.FMNet.ProcessData()

# Initialize 
gatherer = OneComponentGatherData(archive_dir, processing_function)
infilename_roots = ['currentEarthquakeArrivalInformation', 'currentBlastArrivalInformation', 'historicalEarthquakeArrivalInformation']
outfilename_roots = ['current_earthquake_catalog', 'current_blast_catalog', 'historical_earthquake_catalog']
for infilename_root, event_type, outfilename_root in zip(infilename_roots, ['le', 'qb', 'le'], outfilename_roots):
    print(f'loading {outfilename_root}...')
    filename_3c = os.path.join(catalog_dir, infilename_root, '3C.csv')
    filename_1c = os.path.join(catalog_dir, infilename_root, '1C.csv')
    outfilename_root = os.path.join(output_dir, outfilename_root)
    drop_down = False
    if event_type == "qb":
        drop_down = True
    gatherer.gather_waveforms(filename_3c, filename_1c, outfilename_root, event_type, is_first_motion_data=is_first_motion_data, drop_down=drop_down)
    
gatherer.close()
