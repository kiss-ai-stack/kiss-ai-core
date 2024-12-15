import subprocess
import sys

from kiss_ai_stack.core.utilities.logger import LOG


def install_package(package_name: str, upgrade: bool = False) -> None:
    """
    Programmatically install a Python package using pip.

    :param package_name: Name of the package to install.
    :param upgrade: Whether to upgrade the package to the latest version. Defaults to False.

    :raises subprocess.CalledProcessError: If the pip command fails.
    """
    try:
        command = [sys.executable, '-m', 'pip', 'install', package_name]
        LOG.info(f'Executing `{' '.join(command)}`')
        if upgrade:
            command.append('--upgrade')

        subprocess.check_call(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        LOG.info(f'Package \'{package_name}\' installed successfully.')

    except subprocess.CalledProcessError as e:
        LOG.error(f'Failed to install package \'{package_name}\'. Error: {e}')
        raise
