#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from ..config.logger import getLogger, setupLocalLogging
from datetime import datetime
import gzip
import json
import os
import os.path
import pickle
from google.cloud import storage
import google
import google.cloud
import google.cloud.exceptions
from ..components.utils.retry import autoretry
from ..config.config import getSharedGCSStorageClient


def getDataFormatAndCompressionForClass(modelClass, config, overrideSaveFormat=None, overrideCompression=None):
    className = modelClass.__name__

    if isinstance(config.data_serialization_method, str):
        dataFormat = config.data_serialization_method
    else:
        if className in config.data_serialization_method:
            dataFormat = config.data_serialization_method[className]
        else:
            dataFormat = config.data_serialization_method['default']

    if overrideSaveFormat is not None:
        dataFormat = overrideSaveFormat

    if isinstance(config.data_compress_level, int):
        compression = config.data_compress_level
    else:
        if className in config.data_compress_level:
            compression = config.data_compress_level[className]
        else:
            compression = config.data_compress_level['default']

    if overrideCompression is not None:
        compression = overrideCompression

    return dataFormat, compression

@autoretry()
def saveObjectToDisk(targetObject, folder, config, overrideSaveFormat=None, overrideCompression=None):
    dataFormat, compression = getDataFormatAndCompressionForClass(type(targetObject), config, overrideSaveFormat, overrideCompression)

    if dataFormat == "pickle":
        if compression == 0:
            fileName = str(targetObject.id) + ".pickle"
            config.saveKwolaFileData(folder, fileName, pickle.dumps(targetObject, protocol=pickle.HIGHEST_PROTOCOL))
        else:
            fileName = str(targetObject.id) + ".pickle.gz"
            data = pickle.dumps(targetObject, protocol=pickle.HIGHEST_PROTOCOL)
            config.saveKwolaFileData(folder, fileName, gzip.compress(data, compresslevel=compression))

    elif dataFormat == "json":
        if compression == 0:
            fileName = str(targetObject.id) + ".json"
            config.saveKwolaFileData(folder, fileName, bytes(targetObject.to_json(indent=4), 'utf8'))
        else:
            fileName = str(targetObject.id) + ".json.gz"
            config.saveKwolaFileData(folder, fileName, gzip.compress(bytes(targetObject.to_json(indent=4), "utf8"), compresslevel=compression))

    elif dataFormat == "mongo":
        targetObject.save(validate=False)

    elif dataFormat == "gcs":
        fileName = f"{targetObject.id}.json.gz"
        data = gzip.compress(bytes(targetObject.to_json(indent=4), "utf8"), compresslevel=compression)
        config.saveKwolaFileData(folder, fileName, data, useCacheBucket=False)

@autoretry()
def loadObjectFromDisk(modelClass, id, folder, config, printErrorOnFailure=True, applicationId=None):
    try:
        dataFormat, compression = getDataFormatAndCompressionForClass(modelClass, config)

        if dataFormat == "mongo":
            return modelClass.objects(id=id).first()
        elif dataFormat == "gcs":
            if applicationId is None:
                raise RuntimeError("Can't load object from google cloud storage without an applicationId, which is used to indicate the bucket.")

            fileName = f"{id}.json.gz"
            data = config.loadKwolaFileData(folder, fileName, useCacheBucket=False)
            object = modelClass.from_json(str(gzip.decompress(data), "utf8"))
            return object

        object = None
        pickleFileName = str(id) + ".pickle"
        gzipPickleFileName = str(id) + ".pickle.gz"
        jsonFileName = str(id) + ".json"
        gzipJsonFileName = str(id) + ".json.gz"

        # We try loading the data using multiple formats - this allows the format to be changed mid run and the data will still load.
        def tryPickle():
            nonlocal object
            if object is None:
                object = config.loadKwolaFileData(folder, pickleFileName, printErrorOnFailure=printErrorOnFailure)
                if object is not None:
                    object = pickle.loads(object)

        def tryPickleGzip():
            nonlocal object
            if object is None:
                object = config.loadKwolaFileData(folder, gzipPickleFileName, printErrorOnFailure=printErrorOnFailure)
                if object is not None:
                    object = pickle.loads(gzip.decompress(object))

        def tryJson():
            nonlocal object
            if object is None:
                object = config.loadKwolaFileData(folder, jsonFileName, printErrorOnFailure=printErrorOnFailure)
                if object is not None:
                    object = modelClass.from_json(object)

        def tryJsonGzip():
            nonlocal object
            if object is None:
                object = config.loadKwolaFileData(folder, gzipJsonFileName, printErrorOnFailure=printErrorOnFailure)
                if object is not None:
                    object = modelClass.from_json(str(gzip.decompress(object), "utf8"))

        if dataFormat == 'pickle':
            if compression > 0:
                formatOrder = [tryPickleGzip, tryPickle, tryJson, tryJsonGzip]
            else:
                formatOrder = [tryPickle, tryPickleGzip, tryJson, tryJsonGzip]
        else:
            if compression > 0:
                formatOrder = [tryJsonGzip, tryJson, tryPickleGzip, tryPickle]
            else:
                formatOrder = [tryJson, tryJsonGzip, tryPickleGzip, tryPickle]

        for func in formatOrder:
            func()

        if object is None:
            if printErrorOnFailure:
                getLogger().info(f"Error: Failed to load object. File not found. Tried: {pickleFileName}, {gzipPickleFileName}, {jsonFileName}, and {gzipJsonFileName}")
            return None

        return object
    except json.JSONDecodeError:
        if printErrorOnFailure:
            getLogger().info(f"Error: Failed to load object {id}. Bad JSON. Usually implies the file failed to write. "
                                  "Sometimes this occurs if you kill the process while it is running. If this occurs "
                                  "during normal operations without interruption, that would indicate a bug.")
        return
    except EOFError:
        if printErrorOnFailure:
            getLogger().info(f"Error: Failed to load object {id}. Bad pickle file. Usually implies the file failed to write. "
                             "Sometimes this occurs if you kill the process while it is running. If this occurs "
                             "during normal operations without interruption, that would indicate a bug.")
        return
    except FileNotFoundError:
        if printErrorOnFailure:
            getLogger().info(f"Error: Failed to load object {id}. File not found. Usually implies the file failed to write. "
                                  "Sometimes this occurs if you kill the process while it is running. If this occurs "
                                  "during normal operations without interruption, that would indicate a bug.")
        return
    except google.cloud.exceptions.NotFound:
        if printErrorOnFailure:
            getLogger().info(f"Error: Failed to load object {id}. Google cloud storage file not found. Usually implies the file failed to write. "
                                  "Sometimes this occurs if you kill the process while it is running. If this occurs "
                                  "during normal operations without interruption, that would indicate a bug.")
        return
