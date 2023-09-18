import h5py
import numpy as np

pref = "/home/armstrong/Research/constant_boxcar_widths_LG/oneCompPDetector_mew/uuss_data"
split_type = "train"

org_file = f"{pref}/combined.{split_type}.10s.1dup.h5"
mew_file = f"{pref}/{split_type}P.10s.1dup_synthetic_multievent_waveforms.h5"
new_file = f"{pref}/combined.{split_type}.10s.1dup.MEW.h5"

print(org_file)
print(mew_file)

with h5py.File(mew_file, "r") as hf:
    print(hf.keys())
    print("Mew file:", hf["X"].shape, hf["Y"].shape, hf["Pick_index"].shape, hf["Pick_index2"].shape)
    mew_X = hf["X"][:]
    mew_Y = hf["Y"][:]
    mew_T = hf["Pick_index"][:]
    mew_T2 = hf["Pick_index2"][:]

with h5py.File(org_file, "r") as hf:
    print("UUSS file:", hf["X"].shape, hf["Y"].shape, hf["Pick_index"].shape)
    org_X = hf["X"][:]
    org_Y = hf["Y"][:]
    org_T = hf["Pick_index"][:]
    
with h5py.File(new_file, "w") as hf:
    hf.create_dataset("X", data=np.concatenate([mew_X, org_X], axis=0))
    hf.create_dataset("Y", data=np.concatenate([mew_Y, org_Y], axis=0))
    hf.create_dataset("Pick_index", data=np.concatenate([mew_T, org_T], axis=0))
    hf.create_dataset("Pick_index2", data=mew_T2)
    print(hf["X"].shape, hf["Y"].shape, hf["Pick_index"].shape, hf["Pick_index2"].shape)

