import pickle


def read_data(data_path: str) :
    with open(data_path, "br") as f:
        return pickle.load(f)