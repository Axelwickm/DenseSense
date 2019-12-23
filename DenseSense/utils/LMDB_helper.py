from DenseSense.utils.ThingSerializer import ThingSerializer

import lmdb
import os


class LMDB_helper:

    # Mode: r=readonly, a=append, o=override
    def __init__(self, mode="r", path="./data/", verbose=True):
        self.verbose = verbose
        if self.verbose:
            print("Initiating LMDB_helper:\n\tmode = {}\n\tpath = {}\n".format(mode, path))
        self.path = path
        self.mode = mode
        self.databases = {}

    def get(self, Type, key): # TODO: None-key to get whole database at once
        TN = Type.__name__
        if TN not in self.databases:
            self._openDB(TN)
        key = str(key).encode('ascii')
        with self.databases[TN].begin(buffers=True) as t:
            serialized = t.get(key)

        if serialized is None:
            return None

        data = ThingSerializer.decode(serialized)

        if self.verbose:
            print("Loaded LMDB data {} : {}".format(TN, key))
        return data

    def save(self, Type, key, data):
        assert self.mode not in ["r", "read", "readonly"],\
            "LMDB_helper in readonly mode"
        TN = Type.__name__
        if TN not in self.databases:
            self._openDB(TN)
        key = str(key).encode('ascii')

        if self.mode in ["a", "append"]:
            existing = self.get(Type, key)
            if existing is not None:
                return False
            del existing

        if self.verbose:
            print("Saving LMDB data {} : {}".format(TN, key))

        serialized = ThingSerializer.encode(data)
        with self.databases[TN].begin(write=True, buffers=True) as t:
            t.put(key, serialized)

        return True

    def _openDB(self, name):
        if self.verbose:
            print("Opening LMDB database {} in mode {}".format(name, self.mode))
        path = os.path.join(self.path, name)
        if self.mode in ["r", "read", "readonly"]:
            db = lmdb.open(path, readonly=True, create=False)
            self.databases[name] = db
            # TODO: check if opened

        elif self.mode in ["o", "override", "a", "append"]:
            db = lmdb.open(path, readonly=False, create=True)
            # TODO: maybe set permissions
            db.set_mapsize(1028 * 1028 * 1028 * 20)  # 20 GB FIXME: set from outside
            self.databases[name] = db

