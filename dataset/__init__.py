import os
import hashlib
import requests
import shutil
import sys
import tarfile

BASE_URL = 'http://ufldl.stanford.edu/housenumbers'
TRAIN_DATA = 'train.tar.gz'
TEST_DATA = 'test.tar.gz'
EXTRA_DATA= 'extra.tar.gz'
ALL_DATA = [TRAIN_DATA, TEST_DATA, EXTRA_DATA]

MD5 = {
    'extra.tar.gz': '606f41243d71ca4d5fe66dbaf1f02bee',
    'test.tar.gz': '790d9c8d42f1fcbd219b59956c853a81',
    'train.tar.gz': 'a649f4cb15c35520e8a8c342d4c0005a',
}


def show_log(s):
    print(s, end='')
    sys.stdout.flush()

class HouseNumbers:

    def __init__(self):
        cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(cur_dir, '__data__')

    def ensure_data_dir(self):
        """Create data dir."""
        if not os.path.exists(self.data_dir):
            show_log('Data dir %s not exist, creating...' % self.data_dir)
            os.makedirs(self.data_dir)
            show_log('done.\n')

    def md5checksum(self, file):
        """Check file MD5 sum."""
        hash_md5 = hashlib.md5()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def ensure_datafiles(self):
        """Download data files."""
        self.ensure_data_dir()

        for file in ALL_DATA:
            file_path = os.path.join(self.data_dir, file)

            if os.path.exists(file_path):
                # Check if MD5 matches
                show_log('Checking MD5 of %s...' % file)
                md5sum = self.md5checksum(file_path)
                if md5sum != MD5.get(file):
                    show_log('FAILED, remove for redownload\n')
                    os.remove(file_path)
                else:
                    show_log('done.\n')
                    continue

            if not os.path.exists(file_path):
                file_url = '%s/%s' % (BASE_URL, file)
                show_log('%s not found, downloading from %s...' % (file, file_url))
                r = requests.get(file_url, stream=True)
                if r.status_code == 200:
                    with open(file_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                    show_log('done.\n')
                else:
                    show_log('FAILED.\n')
                    return

    def extract_data(self):
        """Extract tar.gz data to data dir."""
        datadirs = [f.replace('.tar.gz', '') for f in ALL_DATA]
        if all(os.path.exists(os.path.join(self.data_dir, d)) for d in datadirs):
            return

        show_log('Some data dirs not exist.\n')
        self.ensure_datafiles()
        for file in ALL_DATA:
            mat = os.path.join(self.data_dir, file.replace('.tar.gz', ''), 'digitStruct.mat')
            if not os.path.exists(mat):
                show_log('Extracting %s...' % file)
                tar = tarfile.open(os.path.join(self.data_dir, file))
                tar.extractall(path=self.data_dir)
                show_log('done.\n')

