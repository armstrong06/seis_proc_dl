import sys
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/deepLearning/apply/np4_build')
import pyuussmlmodels as uuss

def processes_three_component(signalE, signalN, signalZ, dt):
    processing_function = uuss.ThreeComponentPicker.ZRUNet.ProcessData()
    processing_function = uuss.ThreeComponentPicker.ZCNN.ProcessData()

    processed_signal_E = processing_function.process_waveform(signalE, dt)
    processed_signal_N = processing_function.process_waveform(signalN, dt)
    processed_signal_Z = processing_function.process_waveform(signalZ, dt)



def process_one_component(signalZ, sampling_rate):
    processing_function = uuss.OneComponentPicker.ZCNN.ProcessData()
    # Does this function exist?
    processing_function = uuss.OneComponentPicker.ZRUNet.ProcessData()