from ..text import text_to_sequence


def get_data(path, text_cleaners):
    with open(path, "r") as f:
        tests = f.readlines()
    data_list = list(text_to_sequence(test[:-1], text_cleaners) for test in tests)

    return data_list

