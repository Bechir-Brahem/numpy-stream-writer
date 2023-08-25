import shutil
import zipfile
import io
import typing

import numpy as np
from decorator import timing


class IncrementalNpzWriter:
    """
    Write data to npz file incrementally rather than compute all and write
    once, as in ``np.save``. This class can be used with ``contextlib.closing``
    to ensure closed after usage.
    """

    def __init__(self, tofile: str, mode: str = 'x', compress_file=False):
        """
        :param tofile: the ``npz`` file to write
        :param mode: must be one of {'x', 'w', 'a'}. See
               https://docs.python.org/3/library/zipfile.html for detail
        """
        assert mode in 'xwa', str(mode)
        self.compression = zipfile.ZIP_DEFLATED if compress_file else zipfile.ZIP_STORED
        self.tofile = tofile
        self.mode = mode
        self.file = None

    def openFile(self):
        self.file = zipfile.ZipFile(self.tofile, mode=self.mode, compression=self.compression)

    def __enter__(self):
        self.openFile()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @timing(print_args=False)
    def write(self, key: str, data: typing.Union[np.ndarray, bytes],
              ) -> None:
        """
        Same as ``self.write`` but overwrite existing data of name ``key``.

        :param key: the name of data to write
        :param data: the data
        """
        key += '.npy'
        with io.BytesIO() as cbuf:
            np.save(cbuf, data)
            cbuf.seek(0)
            with self.file.open(key, mode=self.mode, force_zip64=True) as outfile:
                shutil.copyfileobj(cbuf, outfile)

    def close(self):
        if self.tofile is not None:
            self.file.close()
            self.tofile = None


@timing(print_args=False)
def save_with_numpy(a):
    np.savez('test2.npz', a=a)


if __name__ == "__main__":
    a, b, c = [np.random.rand(100, 400, 400)] * 3
    with IncrementalNpzWriter('test.npz', 'w') as npzWriter:
        npzWriter.write("a", a)

    save_with_numpy(a)
