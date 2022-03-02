from data_processing.gatherdata import ThreeComponentUUSS
import pyuussmlmodels as uuss
import os 

archive_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/archives/'
catalog_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/data/'
output_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/firstMotion'
phase_type = "P"

processing_function = uuss.ThreeComponentPicker.ZRUNet.ProcessData()

# Initialize 
gatherer = ThreeComponentUUSS(archive_dir, processing_function)
infilename_roots = ['currentEarthquakeArrivalInformation', 'currentBlastArrivalInformation', 'historicalEarthquakeArrivalInformation']
outfilename_roots = ['current_earthquake_catalog', 'current_blast_catalog', 'historical_earthquake_catalog']
for infilename_root, event_type, outfilename_root in zip(infilename_roots, ['le', 'qb', 'le'], outfilename_roots):
    print(f'loading {outfilename_root}...')
    filename = os.path.join(catalog_dir, infilename_root, '3C.csv')
    outfilename_root = os.path.join(output_dir, phase_type, "_", outfilename_root)
    gatherer.process_and_save_waveforms(filename, phase_type, outfilename_root, event_type)
    
gatherer.close()
