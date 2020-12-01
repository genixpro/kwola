#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


from ..config.logger import getLogger, setupLocalLogging
from .LockedFile import LockedFile
from datetime import datetime
import gzip
import json
import os
import os.path
import pickle
from google.cloud import storage
import google
import google.cloud
from ..components.utils.retry import autoretry
from ..components.utils.file import getSharedGCSStorageClient


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

    openFileFunc = LockedFile
    if not config.data_enable_local_file_locking:
        openFileFunc = open

    if dataFormat == "pickle":
        if compression == 0:
            fileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(targetObject.id) + ".pickle")
            with openFileFunc(fileName, 'wb') as f:
                pickle.dump(targetObject, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            fileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(targetObject.id) + ".pickle.gz")
            with openFileFunc(fileName, 'wb') as f:
                data = pickle.dumps(targetObject, protocol=pickle.HIGHEST_PROTOCOL)
                f.write(gzip.compress(data, compresslevel=compression))

    elif dataFormat == "json":
        if compression == 0:
            fileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(targetObject.id) + ".json")
            with openFileFunc(fileName, 'wt') as f:
                f.write(targetObject.to_json(indent=4))
        else:
            fileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(targetObject.id) + ".json.gz")
            with openFileFunc(fileName, 'wb') as f:
                f.write(gzip.compress(bytes(targetObject.to_json(indent=4), "utf8"), compresslevel=compression))

    elif dataFormat == "mongo":
        targetObject.save()

    elif dataFormat == "gcs":
        storageClient = getSharedGCSStorageClient()
        applicationStorageBucket = storage.Bucket(storageClient, "kwola-testing-run-data-" + targetObject.applicationId)
        objectPath = os.path.join(folder, f"{targetObject.id}.json.gz")
        objectBlob = storage.Blob(objectPath, applicationStorageBucket)
        data = gzip.compress(bytes(targetObject.to_json(indent=4), "utf8"), compresslevel=compression)
        objectBlob.upload_from_string(data)

@autoretry()
def loadObjectFromDisk(modelClass, id, folder, config, printErrorOnFailure=True, applicationId=None):
    openFileFunc = LockedFile
    if not config.data_enable_local_file_locking:
        openFileFunc = open

    try:
        dataFormat, compression = getDataFormatAndCompressionForClass(modelClass, config)

        if dataFormat == "mongo":
            return modelClass.objects(id=id).first()
        elif dataFormat == "gcs":
            if applicationId is None:
                raise RuntimeError("Can't load object from google cloud storage without an applicationId, which is used to indicate the bucket.")

            storageClient = getSharedGCSStorageClient()
            applicationStorageBucket = storage.Bucket(storageClient, "kwola-testing-run-data-" + applicationId)
            objectPath = os.path.join(folder, f"{id}.json.gz")
            objectBlob = storage.Blob(objectPath, applicationStorageBucket)
            data = objectBlob.download_as_string()
            object = modelClass.from_json(str(gzip.decompress(data), "utf8"))
            return object

        object = None
        pickleFileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(id) + ".pickle")
        gzipPickleFileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(id) + ".pickle.gz")
        jsonFileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(id) + ".json")
        gzipJsonFileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(id) + ".json.gz")

        # We try loading the data using multiple formats - this allows the format to be changed mid run and the data will still load.
        def tryPickle():
            nonlocal object
            if object is None:
                if os.path.exists(pickleFileName):
                    with openFileFunc(pickleFileName, 'rb') as f:
                        object = pickle.load(f)

        def tryPickleGzip():
            nonlocal object
            if object is None:
                if os.path.exists(gzipPickleFileName):
                    with openFileFunc(gzipPickleFileName, 'rb') as f:
                        object = pickle.loads(gzip.decompress(f.read()))

        def tryJson():
            nonlocal object
            if object is None:
                if os.path.exists(jsonFileName):
                    with openFileFunc(jsonFileName, 'rt') as f:
                        object = modelClass.from_json(f.read())

        def tryJsonGzip():
            nonlocal object
            if object is None:
                if os.path.exists(gzipJsonFileName):
                    with openFileFunc(gzipJsonFileName, 'rb') as f:
                        object = modelClass.from_json(str(gzip.decompress(f.read()), "utf8"))

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
