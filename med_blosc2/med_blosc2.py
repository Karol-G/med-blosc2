from copy import deepcopy
import numpy as np
import blosc2
import math
from typing import Dict, Optional, Union, List, Tuple
from pathlib import Path
import json
import os

MED_BLOSC2_SUFFIX = "mb2nd"
MED_BLOSC2_VERSION = "v0"
MED_BLOSC2_DEFAULT_PATCH_SIZE = 192


class MedBlosc2:
    def __init__(self,
                 array: Union[np.ndarray, str, Path],
                 spacing: Optional[Union[List, Tuple, np.ndarray]] = None,
                 origin: Optional[Union[List, Tuple, np.ndarray]] = None,
                 direction: Optional[Union[List, Tuple, np.ndarray]] = None,
                 header: Optional[Dict] = None,
                 is_seg: Optional[bool] = None,
                 mmap: bool = False,
                 num_threads: int = 1,
                 mode: str = 'r',
                 copy: Optional['MedBlosc2'] = None) -> None:
        """Initializes a MedBlosc2 instance.

        The MedBlosc2 file format (".mb2nd") is a Blosc2-compressed container
        with standardized metadata support for N-dimensional medical images.

        Args:
            array (Union[np.ndarray, str, Path]): Input data or file path. Use
                a numpy ndarray for in-memory arrays. Use a string or Path to
                load a ".b2nd" or ".mb2nd" file.
            spacing (Optional[Union[List, Tuple, np.ndarray]]): Spacing per
                axis. Provide a list/tuple/ndarray with length equal to the
                number of dimensions (e.g., [sx, sy, sz]).
            origin (Optional[Union[List, Tuple, np.ndarray]]): Origin per axis.
                Provide a list/tuple/ndarray with length equal to the number of
                dimensions.
            direction (Optional[Union[List, Tuple, np.ndarray]]): Direction
                cosine matrix. Provide a 2D list/tuple/ndarray with shape
                (ndims, ndims).
            header (Optional[Dict]): Free-form metadata dictionary. Must be
                JSON-serializable when saving.
            is_seg (Optional[bool]): Whether the image is a segmentation mask.
            mmap (bool): Whether to keep the loaded array memory-mapped when
                loading from disk. If true, MedBlosc2.array will be an blosc2.ndarray.NDArray, else np.ndarray.
            num_threads (int): Number of threads for Blosc2 operations.
            mode (str): Blosc2 open mode
                - 'r': read-only, must exist (Default)
                - 'a': read/write, create if doesn't exist (Currently not supported)
                - 'w': create, overwrite if it exists (Currently not supported)
            copy (Optional[MedBlosc2]): Another MedBlosc2 instance to copy
                metadata fields from.
        """
        if isinstance(array, (str, Path)) and (spacing is not None or origin is not None or direction is not None or header is not None or is_seg is not None or copy is not None):
            raise RuntimeError("Spacing, origin, direction, header, is_seg or copy cannot be set if array is a string to load an image.")
        
        if mode != 'r':
            raise NotImplementedError("Currently 'r' is the only supported blosc2 open mode.")
        
        self.spacing, self.origin, self.direction, self.chunk_size, self.block_size, self.patch_size, self._med_blosc2_version = None, None, None, None, None, None, None
        if isinstance(array, (str, Path)):
            array, spacing, origin, direction, header, is_seg, self.chunk_size, self.block_size, self.patch_size, self._med_blosc2_version = self._load(array, num_threads, mode, mmap)

        self.array = array

        # Validate spacing: Must be None or an array-like of same size as image dimensions
        self._has_spacing = False
        if spacing is not None:
            if not isinstance(spacing, (list, tuple, np.ndarray)):
                raise ValueError("Spacing must be either None, list, tuple, or np.ndarray.")
            spacing = np.array(spacing).astype(float)
            if not (spacing.shape == (self.ndims,)):
                raise ValueError(f"The size of the spacing must be the same as the number of image dimensions.")
            self._has_spacing = True
            self.spacing = [float(value) for value in spacing]

        # Validate origin: Must be None or an array-like of same size as image dimensions
        self._has_origin = False
        if origin is not None:
            if not isinstance(origin, (list, tuple, np.ndarray)):
                raise ValueError("Origin must be either None, list, tuple, or np.ndarray.")
            origin = np.array(origin).astype(float)
            if not (origin.shape == (self.ndims,)):
                raise ValueError(f"The size of the origin must be the same as the number of image dimensions.")
            self._has_origin = True
            self.origin = [float(value) for value in origin]

        # Validate direction: Must be None or an array-like with shape (2, 2), (3, 3) or (4, 4) depending on the array dimensionality
        self._has_direction = False
        if direction is not None:
            if not isinstance(direction, (list, tuple, np.ndarray)):
                raise ValueError("Direction must be either None, list, tuple or np.ndarray.")
            direction = np.array(direction).astype(float)
            if not (direction.shape == (self.ndims, self.ndims)):
                raise ValueError(f"The shape of direction must be ({self.ndims}, {self.ndims}).")
            self._has_direction = True
            self.direction = [[float(value) for value in values] for values in direction]

        # Validate header: Must be None or a dictionary
        self._has_header = False
        if header is not None:
            if not isinstance(header, dict):
                raise ValueError("Header must be None or a dictionary.")
            self._has_header = True
        self.header = header
        
        # Validate is_seg: Must be None or a boolean
        self._has_is_seg = False
        if is_seg is not None:
            if not isinstance(is_seg, bool):
                raise ValueError("is_seg must be None or a boolean.")
            self._has_is_seg = True
        self.is_seg = is_seg
        
        # If copy is set, copy fields from the other Nifti instance
        if copy is not None:
            self._copy_fields_from(copy)

    @property
    def affine(self) -> np.ndarray:
        """Computes the affine transformation matrix for the image.

        Returns:
            list: Affine matrix with shape (ndims + 1, ndims + 1).
        """
        spacing  = np.array(self.spacing) if self.spacing is not None else np.ones(self.ndims)
        origin  = np.array(self.origin) if self.origin is not None else np.zeros(self.ndims)
        direction = np.array(self.direction) if self.direction is not None else np.eye(self.ndims)
        affine = np.eye(self.ndims+1)
        affine[:self.ndims, :self.ndims] = direction @ np.diag(spacing)
        affine[:self.ndims, self.ndims] = origin
        return affine.tolist()
    
    @property
    def translation(self):
        """Extracts the translation vector from the affine matrix.

        Returns:
            list: Translation vector with length equal to the number of
            dimensions.
        """
        return np.array(self.affine)[:-1, -1].tolist()

    @property
    def scale(self):
        """Extracts the scaling factors from the affine matrix.

        Returns:
            list: Scaling factors per axis with length equal to the number of
            dimensions.
        """
        scales = np.linalg.norm(np.array(self.affine)[:-1, :-1], axis=0)
        return scales.tolist()

    @property
    def rotation(self):
        """Extracts the rotation matrix from the affine matrix.

        Returns:
            list: Rotation matrix with shape (ndims, ndims).
        """
        rotation_matrix = np.array(self.affine)[:-1, :-1] / np.array(self.scale)
        return rotation_matrix.tolist()

    @property
    def shear(self):
        """Computes the shear matrix from the affine matrix.

        Returns:
            list: Shear matrix with shape (ndims, ndims).
        """
        scales = np.array(self.scale)
        rotation_matrix = np.array(self.rotation)
        shearing_matrix = np.dot(rotation_matrix.T, np.array(self.affine)[:-1, :-1]) / scales[:, None]
        return shearing_matrix.tolist()
    
    @property
    def shape(self):
        """Returns the shape of the array.

        Returns:
            tuple: Shape of the underlying array.
        """
        return self.array.shape
    
    @property
    def ndims(self) -> int:
        """Returns the number of dimensions of the image.

        Returns:
            int: Number of dimensions of the image (2D, 3D, or 4D).
        """
        return len(self.array.shape)

    def _copy_fields_from(self, other: 'MedBlosc2'):
        """Copies fields from another MedBlosc2 instance if unset.

        Args:
            other (MedBlosc2): Instance to copy fields from.
        """
        if not self._has_spacing:
            self.spacing = other.spacing
        if not self._has_origin:
            self.origin = other.origin
        if not self._has_direction:
            self.direction = other.direction
        if not self._has_header:
            self.header = other.header
        if not self._has_is_seg:
            self.is_seg = other.is_seg

    def _load(
            self,
            filepath: Union[str, Path], 
            num_threads: int = 1,
            mode: str = 'r',
            mmap: bool = False
        ):
        """Loads a Blosc2-compressed file. Both MedBlosc2 ('.mb2nd') and Blosc2 ('.b2nd') files are supported.

        WARNING:
            MedBlosc2 supports both ".b2nd" and ".mb2nd" files. The MedBlosc2
            format standard and standardized metadata are honored only for
            ".mb2nd". For ".b2nd", metadata is ignored when loading.

        Args:
            filepath (Union[str, Path]): Path to the Blosc2 file to be loaded.
                The filepath needs to have the extension ".b2nd" or ".mb2nd".
            num_threads (int): Number of threads to use for loading the file.
            mode (str): Blosc2 open mode (e.g., "r", "a").
            mmap (bool): Whether to keep the array memory-mapped.

        Returns:
            Tuple[blosc2.ndarray, dict]: Loaded data and its metadata.

        Raises:
            RuntimeError: If the file extension is not ".b2nd" or ".mb2nd".
        """
        if not str(filepath).endswith(".b2nd") and not str(filepath).endswith(f".{MED_BLOSC2_SUFFIX}"):
            raise RuntimeError(f"MedBlosc2 requires '.b2nd' or '.{MED_BLOSC2_SUFFIX}' as extension.")
        blosc2.set_nthreads(num_threads)
        dparams = {'nthreads': num_threads}
        array = blosc2.open(urlpath=str(filepath), dparams=dparams, mode=mode, mmap_mode=mode)
        metadata = dict(array.schunk.meta)
        spacing, origin, direction, header, is_seg, chunk_size, block_size, patch_size, med_blosc2_version = None, None, None, None, None, None, None, None, None
        if str(filepath).endswith(f".{MED_BLOSC2_SUFFIX}"):
            if "med_blosc2" not in metadata:
                raise RuntimeError(f"The header of the .{MED_BLOSC2_SUFFIX} is missing the 'med_blosc2' attribute.")
            spacing = metadata["med_blosc2"]["spacing"]
            origin = metadata["med_blosc2"]["origin"]
            direction = metadata["med_blosc2"]["direction"]
            header = metadata["med_blosc2"]["header"]
            is_seg = metadata["med_blosc2"]["is_seg"]
            chunk_size = metadata["med_blosc2"]["chunk_size"]
            block_size = metadata["med_blosc2"]["block_size"]
            patch_size = metadata["med_blosc2"]["patch_size"]
            med_blosc2_version = metadata["med_blosc2"]["med_blosc2_version"]
        if not mmap:
            array = array[...]   
        return array, spacing, origin, direction, header, is_seg, chunk_size, block_size, patch_size, med_blosc2_version

    def save(
            self,
            filepath: Union[str, Path],
            patch_size: Optional[Union[int, List, Tuple]] = 'default',  # 'default' means that the default of 192 is used. However, if set to 'default', the patch_size will be skipped if self.patch_size is set from a previously loaded MedBlosc2 image. In that case the self.patch_size is used.
            chunk_size: Optional[Union[int, List, Tuple]]= None,
            block_size: Optional[Union[int, List, Tuple]] = None,
            clevel: int = 8,
            codec: blosc2.Codec = blosc2.Codec.ZSTD,
            num_threads: int = 1
        ):
        """Saves the array to a Blosc2-compressed file. Both MedBlosc2 ('.mb2nd') and Blosc2 ('.b2nd') files are supported.

        WARNING:
            MedBlosc2 supports both ".b2nd" and ".mb2nd" files. The MedBlosc2
            format standard and standardized metadata are honored only for
            ".mb2nd". For ".b2nd", metadata is ignored when saving.

        Args:
            filepath (Union[str, Path]): Path to save the file. Must end with
                ".b2nd" or ".mb2nd".
            patch_size (Optional[Union[int, List, Tuple]]): Patch size hint for
                chunk/block optimization. Provide an int for isotropic sizes or
                a list/tuple with length equal to the number of dimensions.
                Use "default" to use the default patch size of 192.
            chunk_size (Optional[Union[int, List, Tuple]]): Explicit chunk size.
                Provide an int or a tuple/list with length equal to the number
                of dimensions, or None to let Blosc2 decide. Ignored when
                patch_size is not None.
            block_size (Optional[Union[int, List, Tuple]]): Explicit block size.
                Provide an int or a tuple/list with length equal to the number
                of dimensions, or None to let Blosc2 decide. Ignored when
                patch_size is not None.
            clevel (int): Compression level from 0 (no compression) to 9
                (maximum compression).
            codec (blosc2.Codec): Compression codec to use.
            num_threads (int): Number of threads to use for saving the file.

        Raises:
            RuntimeError: If the file extension is not ".b2nd" or ".mb2nd".
        """
        if not str(filepath).endswith(".b2nd") and not str(filepath).endswith(f".{MED_BLOSC2_SUFFIX}"):
            raise RuntimeError(f"MedBlosc2 requires '.b2nd' or '.{MED_BLOSC2_SUFFIX}' as extension.")
        blosc2.set_nthreads(num_threads)
        dparams = {'nthreads': num_threads}
        
        if not(patch_size is None or patch_size == "default" or isinstance(patch_size, int) or len(patch_size) == self.ndims):
            raise RuntimeError(f"patch_size must either be None, an integer, or a list of tuples matching the dimensionality of the array.")
        if not(chunk_size is None or isinstance(chunk_size, int) or len(chunk_size) == self.ndims):
            raise RuntimeError(f"chunk_size must either be None, an integer, or a list of tuples matching the dimensionality of the array.")
        if not(block_size is None or isinstance(block_size, int) or len(block_size) == self.ndims):
            raise RuntimeError(f"block_size must either be None, an integer, or a list of tuples matching the dimensionality of the array.")
        if patch_size is not None and patch_size != "default" and (self.ndims == 1 or self.ndims > 3):
            raise NotImplementedError("Chunk and block size optimization based on patch size is only implemented for 2D and 3D images. Please set the chunk and block size manually or set to None for blosc2 to determine a chunk and block size.")
        if patch_size is not None and patch_size != "default" and (chunk_size is not None or block_size is not None):
            raise RuntimeError("patch_size and chunk_size / block_size cannot both be explicitly set.")

        if patch_size == "default" and self.patch_size is not None:  # Use previously loaded patch size, when patch size is not explicitly set and a patch size from a previously loaded image exists
            patch_size = self.patch_size
            self.patch_size = (patch_size,) * self.ndims if isinstance(patch_size, int) else patch_size
            self.chunk_size, self.block_size = self.comp_blosc2_params(self.array.shape, self.patch_size)
        elif patch_size == "default" and self.patch_size is None:  # Use default patch size, when patch size is not explicitly set and no patch size from a previously loaded image exists
            patch_size = MED_BLOSC2_DEFAULT_PATCH_SIZE
            self.patch_size = (patch_size,) * self.ndims if isinstance(patch_size, int) else patch_size
            self.chunk_size, self.block_size = self.comp_blosc2_params(self.array.shape, self.patch_size)
        elif patch_size is None:  # Ignore patch size and let blosc2 auto configure chunk and block size, when patch size is explicitly set to None
            self.patch_size = None
            self.chunk_size = chunk_size
            self.block_size = block_size
        else:  # Use patch size, when patch size is explicitly to a value
            self.patch_size = (patch_size,) * self.ndims if isinstance(patch_size, int) else patch_size
            self.chunk_size, self.block_size = self.comp_blosc2_params(self.array.shape, self.patch_size)
        
        metadata = None
        if str(filepath).endswith(f".{MED_BLOSC2_SUFFIX}"):
            metadata = {"med_blosc2": {}}
            metadata["med_blosc2"]["spacing"] = self.spacing
            metadata["med_blosc2"]["origin"] = self.origin
            metadata["med_blosc2"]["direction"] = self.direction
            metadata["med_blosc2"]["header"] = self.header
            metadata["med_blosc2"]["is_seg"] = self.is_seg
            metadata["med_blosc2"]["chunk_size"] = self.chunk_size
            metadata["med_blosc2"]["block_size"] = self.block_size
            metadata["med_blosc2"]["patch_size"] = self.patch_size
            metadata["med_blosc2"]["med_blosc2_version"] = MED_BLOSC2_VERSION

        if not self._is_serializable(metadata):
            raise RuntimeError("Image header is not serializable.")

        cparams = {'codec': codec, 'clevel': clevel,}
        self.array = np.ascontiguousarray(self.array[...])  # Needs to overwrite self.array to ensure there is no opened blosc2 overwriting itself
        if Path(filepath).is_file():
            os.remove(str(filepath))
        blosc2.asarray(self.array, urlpath=str(filepath), chunks=self.chunk_size, blocks=self.block_size, cparams=cparams, dparams=dparams, mmap_mode='w+', meta=metadata)

    def comp_blosc2_params(
            self,
            image_size: Tuple[int, int, int, int],
            patch_size: Union[Tuple[int, int], Tuple[int, int, int]],
            bytes_per_pixel: int = 4,  # 4 byte are float32
            l1_cache_size_per_core_in_bytes: int = 32768,  # 1 Kibibyte (KiB) = 2^10 Byte;  32 KiB = 32768 Byte
            l3_cache_size_per_core_in_bytes: int = 1441792, # 1 Mibibyte (MiB) = 2^20 Byte = 1.048.576 Byte; 1.375MiB = 1441792 Byte
            safety_factor: float = 0.8  # we dont will the caches to the brim. 0.8 means we target 80% of the caches
        ):
        """
        Computes a recommended block and chunk size for saving arrays with Blosc v2.

        Blosc2 NDIM documentation:
        "Having a second partition allows for greater flexibility in fitting different partitions to different CPU cache levels. 
        Typically, the first partition (also known as chunks) should be sized to fit within the L3 cache, 
        while the second partition (also known as blocks) should be sized to fit within the L2 or L1 caches, 
        depending on whether the priority is compression ratio or speed." 
        (Source: https://www.blosc.org/posts/blosc2-ndim-intro/)

        Our approach is not fully optimized for this yet. 
        Currently, we aim to fit the uncompressed block within the L1 cache, accepting that it might occasionally spill over into L2, which we consider acceptable.

        Note: This configuration is specifically optimized for nnU-Net data loading, where each read operation is performed by a single core, so multi-threading is not an option.

        The default cache values are based on an older Intel 4110 CPU with 32KB L1, 128KB L2, and 1408KB L3 cache per core. 
        We haven't further optimized for modern CPUs with larger caches, as our data must still be compatible with the older systems.

        Args:
            image_size (Tuple[int, int, int, int]): Image shape. Use a 2D, 3D,
                or 4D size; 2D/3D inputs are internally expanded.
            patch_size (Union[Tuple[int, int], Tuple[int, int, int]]): Patch
                size for spatial dimensions. Use a 2-tuple (x, y) or 3-tuple
                (x, y, z).
            bytes_per_pixel (int): Number of bytes per element. Defaults to 4
                for float32.
            l1_cache_size_per_core_in_bytes (int): L1 cache per core in bytes.
            l3_cache_size_per_core_in_bytes (int): L3 cache per core in bytes.
            safety_factor (float): Safety factor to avoid filling caches.

        Returns:
            Tuple[Tuple[int, ...], Tuple[int, ...]]: Recommended chunk size and block size.
        """

        num_squeezes = 0

        if len(image_size) == 2:
            image_size = (1, 1, *image_size)
            num_squeezes = 2

        if len(image_size) == 3:
            image_size = (1, *image_size)
            num_squeezes = 1

        if len(image_size) != 4:
            raise RuntimeError("Image size must be 4D.")
        
        if not (len(patch_size) == 2 or len(patch_size) == 3):
            raise RuntimeError("Patch size must be 2D or 3D.")

        num_channels = image_size[0]
        if len(patch_size) == 2:
            patch_size = [1, *patch_size]
        patch_size = np.array(patch_size)
        block_size = np.array((num_channels, *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size]))

        # shrink the block size until it fits in L1
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
            # pick largest deviation from patch_size that is not 1
            axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
            idx = 0
            picked_axis = axis_order[idx]
            while block_size[picked_axis + 1] == 1 or block_size[picked_axis + 1] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            # now reduce that axis to the next lowest power of 2
            block_size[picked_axis + 1] = 2 ** (max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1))))
            block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

        block_size = np.array([min(i, j) for i, j in zip(image_size, block_size)])

        # note: there is no use extending the chunk size to 3d when we have a 2d patch size! This would unnecessarily
        # load data into L3
        # now tile the blocks into chunks until we hit image_size or the l3 cache per core limit
        chunk_size = deepcopy(block_size)
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
            if patch_size[0] == 1 and all([i == j for i, j in zip(chunk_size[2:], image_size[2:])]):
                break
            if all([i == j for i, j in zip(chunk_size, image_size)]):
                break
            # find axis that deviates from block_size the most
            axis_order = np.argsort(chunk_size[1:] / block_size[1:])
            idx = 0
            picked_axis = axis_order[idx]
            while chunk_size[picked_axis + 1] == image_size[picked_axis + 1] or patch_size[picked_axis] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
            chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
            if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size)]) > 1.5:
                # chunk size should not exceed patch size * 1.5 on average
                chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
                break
        # better safe than sorry
        chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size)]

        block_size = block_size[num_squeezes:]
        chunk_size = chunk_size[num_squeezes:]

        return tuple([int(value) for value in chunk_size]), tuple([int(value) for value in block_size])
    
    def _is_serializable(self, d: dict) -> bool:
        """Checks whether a dictionary is JSON-serializable.

        Args:
            d (dict): Input dictionary to test.

        Returns:
            bool: True when serializable, otherwise False.
        """
        try:
            json.dumps(d)
            return True
        except (TypeError, OverflowError):
            return False
    

if __name__ == '__main__':
    print("Creating array...")
    array = np.random.random((128, 512, 512))
    spacing = (2, 2.5, 4)
    origin = None
    direction = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    header = {"tmp1": "This is an image", "tmp2": 5, "tmp3": {"test1": 16.4587, "test2": [1, 2, 3, 4, 5, 6]}}
    filepath = "tmp.mb2nd"

    print("Initializing image...")
    image = MedBlosc2(array, spacing=spacing, origin=origin, direction=direction, header=header)
    print("Saving image...")
    image.save(filepath)

    print("Loading image...")
    image = MedBlosc2(filepath, mmap=True)
    print("array type: ", type(image.array))
    print("spacing: ", image.spacing)
    print("origin: ", image.origin)
    print("direction: ", image.direction)
    print("header: ", image.header)
    print("chunk_size: ", image.chunk_size)
    print("block_size: ", image.block_size)
    print("patch_size: ", image.patch_size)
    print("version: ", image._med_blosc2_version)

    if Path(filepath).is_file():
        os.remove(filepath)
