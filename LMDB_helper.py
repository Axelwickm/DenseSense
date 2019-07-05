import copy
import io
import json
import os
import sys
from collections import Mapping, Sequence, Set
from stat import *

import cv2
import numpy as np

import lmdb

try:
    from six import string_types, iteritems
except ImportError:
    string_types = (str, unicode) if str is bytes else (str, bytes)
    iteritems = lambda mapping: getattr(mapping, 'iteritems', mapping.items)()


# Class for dynamically accesing photodata

class PhotoData(object):
    def __init__(self, path):
        self.env = lmdb.open(
            path, map_size=2**36, readonly=True, lock=False
        )
    
    def __iter__(self):
        with self.env.begin() as t:
            with t.cursor() as c:
                for key, value in c:
                    yield key, value
        
    def __getitem__(self, index):
        key = str(index).encode('ascii')
        with self.env.begin() as t:
            data = t.get(key)
        if not data:
            return None
        with io.BytesIO(data) as f:
            return cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_COLOR)
    
    def __contains__(self, index):
        key = str(index).encode('ascii')

        with self.env.begin() as t:
            data = t.get(key)
        if not data:
            return False
        return True

    def __len__(self):
        return self.env.stat()['entries']


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class LMDB_helper(object):

    auxiliaryDB = None
    loadedData = {}
    dataGenerators = {}
    
    def __init__(self, filepath, saveToFile, override, loadedLimitMB):
        self.filepath = filepath
        self.saveToFile = saveToFile
        self.override = override
        self.loadedLimitMB = loadedLimitMB

        if self.saveToFile:
            # Create/open database with write permissions
            self.auxiliaryDB = lmdb.open(filepath, readonly = False, create = True)
            # You get permissions! And YOU get permissions! EVERYONE GETS PERMISSIONS!!!
            os.chmod(filepath, \
                        S_IRUSR | S_IWUSR | S_IXUSR | \
                        S_IRGRP | S_IWGRP | S_IXGRP | \
                        S_IROTH | S_IWOTH | S_IXOTH )
            
            for root, dirs, files in os.walk(filepath):  
                for file in files:
                    fname = os.path.join(root, file)
                    os.chmod(fname, \
                        S_IRUSR | S_IWUSR | S_IXUSR | \
                        S_IRGRP | S_IWGRP | S_IXGRP | \
                        S_IROTH | S_IWOTH | S_IXOTH )
            self.auxiliaryDB.set_mapsize(1028*1028*1028*135) # About 135 GB
        else:
            self.auxiliaryDB = lmdb.open(filepath, readonly = True, create = False)
    
    def getFromAux(self, key, buffers = False):
        if self.auxiliaryDB != None:
            key = str(key).encode('ascii')
            with self.auxiliaryDB.begin(buffers = buffers) as t:
                data = t.get(key)
            if data is not None:
                print("Loading data from auxillaryDB: "+key)
                return data


    def writeToAux(self, key, val, buffers = False):
        if self.auxiliaryDB != None and self.saveToFile:
            print("Writing to auxiliaryDB: "+key)
            key = str(key).encode('ascii')
            with self.auxiliaryDB.begin(write = True, buffers = buffers) as t:
                t.put(key, val)
    

    def saveData(self, dataset, key, value, encoding="string"):
        key = dataset+"_"+str(key)

        if not self.saveToFile:
            return

        def save(k, v, t):
            if t == "string":
                self.writeToAux(k, v)
            if t == "nparray":
                self.writeToAux(k, json.dumps(v, cls=NumpyEncoder))
            if t == "ndarray":
                self.writeToAux(k, np.getbuffer(v), True)

        def objwalk(obj, values, auxKey, path=(), memo=None):
            if memo is None:
                memo = set()
            if isinstance(obj, Mapping):
                if id(obj) not in memo:
                    memo.add(id(obj)) 
                    for key, value in iteritems(obj):
                        for child in objwalk(value, values, auxKey, path + (key,), memo):
                            yield child
            elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, string_types):
                if id(obj) not in memo:
                    memo.add(id(obj))
                    for index, value in enumerate(obj):
                        for child in objwalk(value, values, auxKey, path + (index,), memo):
                            yield child
            elif obj != "":
                for p in path[:-1]:
                    values = values[p]
                strPath = list(path)
                for i in range(len(path)):
                    strPath[i] = str(strPath[i])
                kk = "|".join(strPath)
                save(auxKey+"."+kk, values[path[-1]], obj)
                if obj in ["nparray","ndarray"]:
                    shape = list(values[path[-1]].shape)
                    dtype = values[path[-1]].dtype.name
                    values[path[-1]] = {"shape":shape, "dtype":dtype}
                else:
                    values[path[-1]] = -1

        if encoding == "string":
            save(key, value, encoding)
        elif encoding == "nparray":
            meta = {
                "type" : "nparray",
                "dtype" : value.dtype.name
            }
            save(key+".META", json.dumps(meta), "string")
            save(key, value, encoding)
        elif encoding == "ndarray":
            meta = {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype" : value.dtype.name
            }
            save(key+".META", json.dumps(meta), "string")
            save(key, value, encoding)
            self.getFromAux(key, True)

        else:
            value = copy.deepcopy(value)
            for _ in objwalk(encoding, value, key):
                pass
            
            meta = {
                "type" : "multi",
                "encoding" : encoding,
                "object" : value
            }
            save(key+".META", json.dumps(meta), "string")
            
    def getData(self, dataset, key):
        key = str(key)

        def auxTypeGet(k, t, meta=None):
            if t == "string":
                return self.getFromAux(k)
            elif t == "nparray":
                entry = self.getFromAux(k)
                entry = json.loads(entry)
                entry = np.asarray(entry, dtype=meta["dtype"])
                return entry
            elif t == "ndarray":
                entry = self.getFromAux(k, True)
                return np.frombuffer(entry, meta["dtype"]).reshape(tuple(meta["shape"])) 
            elif t == "multi":
                for _ in objwalk(meta["encoding"], meta["object"], k):
                    pass
                return meta["object"]
        
        def objwalk(obj, values, auxKey, path=(), memo=None):
            if memo is None:
                memo = set()
            if isinstance(obj, Mapping):
                if id(obj) not in memo:
                    memo.add(id(obj)) 
                    for key, value in iteritems(obj):
                        for child in objwalk(value, values, auxKey, path + (key,), memo):
                            yield child
            elif isinstance(obj, (Sequence, Set)) and not isinstance(obj, string_types):
                if id(obj) not in memo:
                    memo.add(id(obj))
                    for index, value in enumerate(obj):
                        for child in objwalk(value, values, auxKey, path + (index,), memo):
                            yield child
            elif obj != "":
                for p in path[:-1]:
                    values = values[p]
                strPath = list(path)

                for i in range(len(path)):
                    strPath[i] = str(strPath[i])
                kk = "|".join(strPath)

                meta = {}
                if obj in ["nparray", "ndarray"]:
                    meta = copy.copy(values[path[-1]])

                values[path[-1]] = auxTypeGet(auxKey+"."+kk, obj, meta)

        if dataset not in self.loadedData:
            self.loadedData[dataset] = {}

        if key not in self.loadedData[dataset]:
            auxKey = dataset+"_"+str(key)
            entryMeta = self.getFromAux(auxKey+".META")

            if dataset in self.override:
                newData = self.generateData(dataset, key)
                if newData is not None: # This is probably a generator that loads everything into memory at once
                    self.loadedData[dataset][key] = newData
                if key == "":
                    return None

            # Load from aux
            elif entryMeta is not None:
                # There is a metafile, follow its' loading instructions
                entryMeta = json.loads(entryMeta)
                self.loadedData[dataset][key] = auxTypeGet(auxKey, entryMeta["type"], entryMeta)
            else: 
                # No metafile, try just to load the entry as string and return
                entry = self.getFromAux(auxKey)
                if entry is not None:
                    self.loadedData[dataset][key] = entry
                else:
                    # Couldn't load from aux DB it must be created
                    newData = self.generateData(dataset, key)
                    if newData == "NOTHING":
                        return None
                    
                    if newData is not None: # This might be a generator that loads everything into memory all at once
                        self.loadedData[dataset][key] = newData

        return self.loadedData[dataset][key]
    
    def getAllLoaded(self):
        return self.loadedData

    def registerGenerator(self, dataset, cb):
        self.dataGenerators[dataset] = cb

    def generateData(self, dataset, key):
        print("Generating "+dataset+": "+str(key))
        if self.loadedLimitMB*1000000 < sys.getsizeof(self.loadedData):
            if len(self.loadedData[dataset]) != 0:
                print("Unload data from RAM in dataset"+dataset, self.loadedLimitMB, sys.getsizeof(self.loadedData))
                keys = np.asarray(list(self.loadedData[dataset].keys()))
                del self.loadedData[dataset][np.random.choice(keys, size=1)[0]]
        data = self.dataGenerators[dataset](key)
        return data
