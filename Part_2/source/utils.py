import pickle


def save_file(file_name, params):
    with open(file_name, "wb") as f:
        pickle.dump(params, f)


def load_file(file_name):
    with open(file_name, "rb") as f:
        params = pickle.load(f)
    return params
