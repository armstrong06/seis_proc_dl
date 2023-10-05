import h5py

class Write():

    @staticmethod
    def h5py_file(names, data_list, outfile):
        file = h5py.File(outfile, "w")
        for name, data in zip(names, data_list):
            file.create_dataset(name=name, data=data)
        print(f"h5py {outfile} with keys {file.keys()} created.")
        file.close()

class Read():
    @staticmethod
    def hdf5_file(filename):
        file = h5py.File(filename, "r")
        return_data = {}
        for key in file.keys():
            return_data[key] = file[key][:]
        file.close()
        return return_data

