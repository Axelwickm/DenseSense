import pickle
import zlib

# Currently just pickles things,
# but there is room for expansion


class ThingSerializer:
    @staticmethod
    def encode(thing):
        if thing is None:
            return None
        serialized = pickle.dumps(thing)
        serialized = zlib.compress(serialized)
        return serialized

    @staticmethod
    def decode(serialized):
        if serialized is None:
            return None
        serialized = zlib.decompress(serialized)
        thing = pickle.loads(serialized)
        return thing
