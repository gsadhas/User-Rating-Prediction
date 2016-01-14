import os


# make sure the directories exist when specified
def assure_path_exists(path):
    """
    make sure the path exists
    :param path: path to be created
    :return: None
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created directories")
    else:
        print("Already exists")
