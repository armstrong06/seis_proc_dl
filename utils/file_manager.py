import h5py

class Write():

    @staticmethod
    def h5py_file(names, data_list, outfile):
        file = h5py.File(outfile, "w")
        for name, data in zip(names, data_list):
            file.create_dataset(name, data=data)
        print(f"h5py with keys {file.keys()} created.")
        file.close()
