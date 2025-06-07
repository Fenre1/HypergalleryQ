# utils/file_utils.py

import os
import glob
from typing import List

def get_image_files(directory: str) -> List[str]:
    """
    Scans a directory and its subdirectories for image files with specific extensions.
    Args:
        directory (str): The root folder to scan for images.
    Returns:
        List[str]: A list of file paths for all found images.
    """
    # List of image file extensions to search for.
    extensions = ['jpg', 'JPG', 'gif', 'GIF', 'png', 'PNG', 'tif', 'TIF', 'bmp', 'BMP']
    file_paths = []
    for ext in extensions:
        pattern = os.path.join(directory, '**', f'*.{ext}')
        file_paths.extend(glob.glob(pattern, recursive=True))
    return list(set(file_paths))
