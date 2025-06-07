      
import numpy as np
import h5py
from PIL import Image, ImageOps
import os
from typing import List, Tuple, Union, Optional, Dict
import io # Required for in-memory image handling

class HDF5ProjectHandler:
    def __init__(self):
        pass

    def _process_single_thumbnail(self,
                                 input_path: str,
                                 target_size: Tuple[int, int],
                                 save_to_disk_path: Optional[str] = None
                                 ) -> Optional[bytes]:
        """
        Processes one image into a thumbnail.
        Optionally saves it to disk and/or returns its byte representation.
        """
        try:
            image = Image.open(input_path).convert('RGB')
            image.thumbnail(target_size, Image.Resampling.LANCZOS)

            final_thumb = Image.new("RGB", target_size, "black")
            paste_x = (target_size[0] - image.width) // 2
            paste_y = (target_size[1] - image.height) // 2
            final_thumb.paste(image, (paste_x, paste_y))

            if save_to_disk_path:
                final_thumb.save(save_to_disk_path, 'JPEG', quality=90)
                # If only saving to disk and not returning bytes, we could return None here
                # But for flexibility, let's always prepare bytes if not exclusively saving to disk.

            # Prepare bytes for HDF5 storage or if no disk path provided
            img_byte_arr = io.BytesIO()
            final_thumb.save(img_byte_arr, format='JPEG', quality=90)
            return img_byte_arr.getvalue()

        except FileNotFoundError:
            print(f"Warning: Original image not found, skipping thumbnail: {input_path}")
            return None
        except Exception as e:
            print(f"Warning: Could not create thumbnail for {input_path}: {e}")
            return None


    def generate_thumbnails_data(self,
                                image_file_paths: List[str],
                                thumbnail_size: Tuple[int, int] = (100, 100),
                                store_externally: bool = False, # Default OFF
                                external_thumbnail_dir: Optional[str] = None
                               ) -> Union[List[Optional[bytes]], List[Optional[str]]]:
        """
        Generates thumbnail data.
        If store_externally is True, saves .jpg files to external_thumbnail_dir
        and returns relative paths.
        Otherwise (default), returns a list of thumbnail image bytes.

        Args:
            image_file_paths (List[str]): Absolute paths to original images.
            thumbnail_size (Tuple[int, int]): Target size for thumbnails.
            store_externally (bool): If True, save thumbnails as external files.
            external_thumbnail_dir (str, optional): Absolute path to dir for external thumbnails.
                                                 Required if store_externally is True.

        Returns:
            Union[List[Optional[bytes]], List[Optional[str]]]:
                - List of bytes (JPEG data) if store_externally is False.
                - List of relative paths if store_externally is True.
                None for failed thumbnails.
        """
        if store_externally and not external_thumbnail_dir:
            raise ValueError("external_thumbnail_dir must be provided if store_externally is True.")

        output_data = []
        print(f"Generating {len(image_file_paths)} thumbnails...")
        if store_externally:
            os.makedirs(external_thumbnail_dir, exist_ok=True)
            print(f"Thumbnails will be saved externally in: {external_thumbnail_dir}")

        for i, file_path in enumerate(image_file_paths):
            disk_save_path = None
            relative_path_for_hdf5 = None

            if store_externally:
                thumb_filename = f"{i}.jpg"
                disk_save_path = os.path.join(external_thumbnail_dir, thumb_filename)
                # Path stored in HDF5 is relative to HDF5 file's directory
                relative_path_for_hdf5 = os.path.join(os.path.basename(external_thumbnail_dir), thumb_filename).replace('\\', '/')

            thumbnail_bytes = self._process_single_thumbnail(file_path, thumbnail_size, disk_save_path)

            if store_externally:
                if thumbnail_bytes: # if processing was successful
                    output_data.append(relative_path_for_hdf5)
                else:
                    output_data.append(None) # Or an empty string for path
            else: # Store bytes
                output_data.append(thumbnail_bytes)

        print("Thumbnail data generation complete.")
        return output_data


    def save_project_to_hdf5(self,
                             hdf5_file_path: str,
                             image_file_paths: List[str],
                             features: np.ndarray,
                             thumbnail_data: Union[List[Optional[bytes]], List[Optional[str]]],
                             thumbnails_are_embedded: bool = True, # Default True
                             clustering_results: Optional[np.ndarray] = None):
        """
        Saves all project data to an HDF5 file.
        """
        print(f"Saving project data to: {hdf5_file_path}")
        with h5py.File(hdf5_file_path, 'w') as f:
            dt_str = h5py.string_dtype(encoding='utf-8')
            if image_file_paths:
                f.create_dataset('image_file_paths', data=np.array(image_file_paths, dtype=object), dtype=dt_str)

            if features is not None and features.size > 0:
                chunk_shape = (min(100, features.shape[0]), features.shape[1]) if features.ndim == 2 and features.shape[0] > 0 else None
                f.create_dataset('features', data=features, dtype='f4', chunks=chunk_shape)

            f.attrs['thumbnails_are_embedded'] = thumbnails_are_embedded

            if thumbnail_data:
                if thumbnails_are_embedded:
                    # Store as variable-length byte strings (each element is a JPEG image)
                    # Filter out None values before creating dataset
                    valid_thumbnail_bytes = [tb for tb in thumbnail_data if tb is not None]
                    if valid_thumbnail_bytes:
                        # HDF5 requires objects to be of np.void type for variable length binary
                        dt_vlen_bytes = h5py.vlen_dtype(np.uint8)
                        # Create a list of numpy arrays from bytes
                        np_thumbnail_data = [np.frombuffer(b, dtype=np.uint8) for b in valid_thumbnail_bytes]
                        f.create_dataset('thumbnail_data_embedded', data=np_thumbnail_data, dtype=dt_vlen_bytes)
                    else: # Handle case where all thumbnails failed but still embedded
                        f.create_dataset('thumbnail_data_embedded', shape=(0,), dtype=h5py.vlen_dtype(np.uint8))

                else: # Store relative paths
                    valid_thumbnail_paths = [p for p in thumbnail_data if p is not None]
                    if valid_thumbnail_paths:
                        f.create_dataset('thumbnail_relative_paths', data=np.array(valid_thumbnail_paths, dtype=object), dtype=dt_str)
                    else: # Handle case where all paths failed
                         f.create_dataset('thumbnail_relative_paths', shape=(0,), dtype=dt_str)


            if clustering_results is not None and clustering_results.size > 0:
                f.create_dataset('clustering_results', data=clustering_results, dtype='i8')
            else:
                f.create_dataset('clustering_results', data=np.array([], dtype='i8'))

            f.attrs['project_format_version'] = '1.1' # Increment version
        print("Project saved successfully.")


    def load_project_from_hdf5(self, hdf5_file_path: str) -> Dict:
        """
        Loads project data from an HDF5 file.
        """
        print(f"Loading project data from: {hdf5_file_path}")
        data = {
            'image_file_paths': None,
            'features': None,
            'thumbnail_data': None, # Will hold bytes or paths
            'thumbnails_are_embedded': True, # Default assumption
            'clustering_results': None
        }
        if not os.path.exists(hdf5_file_path):
            print(f"Error: Project file not found at {hdf5_file_path}")
            return data

        with h5py.File(hdf5_file_path, 'r') as f:
            if 'image_file_paths' in f:
                data['image_file_paths'] = [path.decode('utf-8') for path in f['image_file_paths'][:]]
            if 'features' in f:
                data['features'] = f['features'][:]

            data['thumbnails_are_embedded'] = f.attrs.get('thumbnails_are_embedded', True) # Default to True for new format

            if data['thumbnails_are_embedded']:
                if 'thumbnail_data_embedded' in f:
                    # Loaded as list of np.ndarray of uint8, convert to bytes
                    data['thumbnail_data'] = [arr.tobytes() for arr in f['thumbnail_data_embedded'][:]]
            else:
                if 'thumbnail_relative_paths' in f:
                    data['thumbnail_data'] = [path.decode('utf-8') for path in f['thumbnail_relative_paths'][:]]

            if 'clustering_results' in f:
                data['clustering_results'] = f['clustering_results'][:]
        print("Project loaded.")
        return data

    