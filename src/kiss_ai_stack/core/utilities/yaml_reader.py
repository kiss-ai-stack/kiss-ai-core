import aiofiles
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

    async def read(self):
        """
        Reads and parses the YAML file asynchronously.

        :return: Parsed data as a Python dictionary
        :raises FileNotFoundError: If the file does not exist
        :raises yaml.YAMLError: If there's an error in the YAML format
        """
        if not self.__file_obj:
            try:
                self.__file_obj = await aiofiles.open(self.__file_path, 'r')
            except FileNotFoundError:
                LOG.error(f'Error: File \'{self.__file_path}\' not found.')
                raise FileNotFoundError(f'File \'{self.__file_path}\' not found.')

        try:
            content = await self.__file_obj.read()
            return yaml.safe_load(content)
        except yaml.YAMLError as e:
            LOG.error(f'Error parsing YAML file: {e}')
            raise

    async def __aenter__(self):
        """
        Enter the runtime context related to this object asynchronously.

        :return: YamlReader instance
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context related to this object asynchronously. Ensures that the file is closed.

        :param exc_type: Exception type (if any)
        :param exc_val: Exception value (if any)
        :param exc_tb: Traceback object (if any)
        """
        if self.__file_obj:
            await self.__file_obj.close()
