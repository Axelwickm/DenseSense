import lmdb
import os
from DenseSense.utils.ThingSerializer import ThingSerializer


class LMDBHelper:

    # Mode: r=readonly, a=append, o=override
    def __init__(self, mode="r", path="./data/", verbose=True, prefix="LMDB_"):
        self.verbose = verbose
        if self.verbose:
            print("Initiating LMDB_helper:\n\tmode = {}\n\tpath = {}\n".format(mode, path))
        self.path = path
        self.mode = mode
        self.databases = {}
        self.prefix = prefix

    def get(self, Type, key):  # TODO: None-key to get whole database at once
        TN = self.prefix + Type.__name__  # Filename
        if TN not in self.databases:      # Check if already opened
            self.openDatabase(TN)

        key = str(key).encode('ascii')
        with self.databases[TN].begin(buffers=True) as t:
            serialized = t.get(key)       # Get data

        if serialized is None:
            return None

        data = ThingSerializer.decode(serialized)  # Deserialize

        if self.verbose:
            print("Loaded LMDB data {} : {}".format(TN, key))
        return data

    def save(self, Type, key, data):
        assert self.mode not in ["r", "read", "readonly"], \
            "LMDB_helper in readonly mode"
        TN = self.prefix + Type.__name__  # Filename
        if TN not in self.databases:      # Check if already opened
            self.openDatabase(TN)

        key = str(key).encode('ascii')

        if self.mode in ["a", "append"]:  # Make sure not to override
            existing = self.get(Type, key)
            if existing is not None:
                return False
            del existing

        if self.verbose:
            print("Saving LMDB data {} : {}".format(TN, key))

        serialized = ThingSerializer.encode(data)  # Serialize
        with self.databases[TN].begin(write=True, buffers=True) as t:
            t.put(key, serialized)                 # Write

        return True

    def openDatabase(self, name, maxSize=1028*1028*1028*10):  # Default max size is 10 GB
        if self.verbose:
            print("Opening LMDB database {} in mode {}".format(name, self.mode))
        path = os.path.join(self.path, name)
        if self.mode in ["r", "read", "readonly"]:
            db = lmdb.open(path, readonly=True, create=False)
            self.databases[name] = db
            # TODO: check if opened

        elif self.mode in ["o", "override", "a", "append"]:
            db = lmdb.open(path, readonly=False, create=True)
            db.set_mapsize(maxSize)
            self.databases[name] = db
