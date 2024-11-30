import yaml

from kiss_ai_stack.core.utilities.logger import LOG


class YamlReader:
    def __init__(self, file_path):
        """
        Initializes the YamlReader with the path to a YAML file.

        :param file_path: Path to the YAML file
        """
        self.__file_path = file_path
        self.__file_obj = None

    def read(self):
        """
        Reads and parses the YAML file.

        :return: Parsed data as a Python dictionary
        :raises FileNotFoundError: If the file does not exist
        :raises yaml.YAMLError: If there's an error in the YAML format
        """
        if not self.__file_obj:
            try:
                self.__file_obj = open(self.__file_path, 'r')
            except FileNotFoundError:
                LOG.error(f'Error: File \'{self.__file_path}\' not found.')
                raise FileNotFoundError(f'File \'{self.__file_path}\' not found.')

        try:
            return yaml.safe_load(self.__file_obj)
        except yaml.YAMLError as e:
            LOG.error(f'Error parsing YAML file: {e}')
            raise

    def __enter__(self):
        """
        Enter the runtime context related to this object.

        :return: YamlReader instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context related to this object. Ensures that the file is closed.

        :param exc_type: Exception type (if any)
        :param exc_val: Exception value (if any)
        :param exc_tb: Traceback object (if any)
        """
        if self.__file_obj:
            self.__file_obj.close()
