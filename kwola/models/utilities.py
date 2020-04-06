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


import os.path
import gzip
from .lockedfile import LockedFile
import json
from datetime import datetime
import pickle



def saveObjectToDisk(targetObject, folder, config):
    if config.data_serialization_method == "pickle":
        if config.data_compress_level == 0:
            fileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(targetObject.id) + ".pickle")
            with LockedFile(fileName, 'wb') as f:
                pickle.dump(targetObject, f)
        else:
            fileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(targetObject.id) + ".pickle.gz")
            with LockedFile(fileName, 'wb') as f:
                data = pickle.dumps(targetObject)
                f.write(gzip.compress(data, compresslevel=config.data_compress_level))

    elif config.data_serialization_method == "json":
        if config.data_compress_level == 0:
            fileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(targetObject.id) + ".json")
            with LockedFile(fileName, 'wt') as f:
                f.write(targetObject.to_json(indent=4))
        else:
            fileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(targetObject.id) + ".json.gz")
            with LockedFile(fileName, 'wb') as f:
                f.write(gzip.compress(bytes(targetObject.to_json(indent=4), "utf8"), compresslevel=config.data_compress_level))



def loadObjectFromDisk(modelClass, id, folder, config):
    try:
        object = None
        pickleFileName = ''
        gzipPickleFileName = ''
        jsonFileName = ''
        gzipJsonFileName = ''

        # Try to load a pickled version first, since its the fastest.
        pickleFileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(id) + ".pickle")
        if os.path.exists(pickleFileName):
            with LockedFile(pickleFileName, 'rb') as f:
                object = pickle.load(f)

        # Next try a gzipped pickled version
        if object is None:
            gzipPickleFileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(id) + ".pickle.gz")
            if os.path.exists(gzipPickleFileName):
                with LockedFile(gzipPickleFileName, 'rb') as f:
                    object = pickle.loads(gzip.decompress(f.read()))

        # Next try to load vanilla json version
        if object is None:
            jsonFileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(id) + ".json")
            if os.path.exists(jsonFileName):
                with LockedFile(jsonFileName, 'rt') as f:
                    object = modelClass.from_json(f.read())

        if object is None:
            # Lastly, try to load a gzipped json version. We do this last since its the slowest.
            gzipJsonFileName = os.path.join(config.getKwolaUserDataDirectory(folder), str(id) + ".json.gz")
            if os.path.exists(gzipJsonFileName):
                with LockedFile(gzipJsonFileName, 'rb') as f:
                    object = modelClass.from_json(str(gzip.decompress(f.read()), "utf8"))

        if object is None:
            print(datetime.now(), "Error: Failed to load object. File not found:", pickleFileName, gzipPickleFileName, jsonFileName, gzipJsonFileName, flush=True)
            return None

        return object
    except json.JSONDecodeError:
        print(datetime.now(), f"Error: Failed to load object {id}. Bad JSON. Usually implies the file failed to write. "
                              "Sometimes this occurs if you kill the process while it is running. If this occurs "
                              "during normal operations without interruption, that would indicate a bug.", flush=True)
        return
    except EOFError:
        print(datetime.now(), f"Error: Failed to load object {id}. Bad pickle file. Usually implies the file failed to write. "
                              "Sometimes this occurs if you kill the process while it is running. If this occurs "
                              "during normal operations without interruption, that would indicate a bug.", flush=True)
        return
