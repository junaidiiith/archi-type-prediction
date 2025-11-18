import os
import fnmatch
from typing import List


def find_files_with_extension(root_dir: str, extension: str) -> List[str]:
    matching_files: List[str] = list()

    # Recursively search for files with the specified extension
    for root, _, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, f'*.{extension}'):
            matching_files.append(os.path.join(root, filename))

    return matching_files

__all__ = ["find_files_with_extension"]