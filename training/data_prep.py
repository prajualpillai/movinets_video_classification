import tensorflow as tf
import remotezip as rz


class DataPrep:
    """
    The class will contain function to perform the following functions:
        1. Download/procure data(in .zip format) from the url provided
        2. Get the classes of the data required
        3. Select a subset of the data for testing purposes
        4. Download the data from the zipped file
        5. Create frames of each video file
        6.
    """

    def __init__(self):
        self.data_path = ""

    def list_files_from_zip(self, path: str) -> list:
        """

        :param path:
        :return:
        """
        files = []
        with rz.RemoteZip(path) as zip:
            for zip_info in zip.infolist():
                files.append(zip_info.filename)
            return files
