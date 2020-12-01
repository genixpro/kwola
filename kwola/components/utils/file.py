from google.cloud import storage
import google
import google.cloud
from ...config.logger import getLogger, setupLocalLogging
from ...components.utils.retry import autoretry

globalStorageClient = None
def getSharedGCSStorageClient():
    global globalStorageClient
    if globalStorageClient is None:
        globalStorageClient = storage.Client()

    return globalStorageClient

@autoretry()
def saveKwolaFileData(filePath, fileData, config):
    if config['data_file_storage_method'] == 'local':
        with open(filePath, 'wb') as f:
            f.write(fileData)
    elif config['data_file_storage_method'] == 'gcs':
        if 'applicationId' not in config or config.applicationId is None:
            raise RuntimeError("Can't load object from google cloud storage without an applicationId, which is used to indicate the bucket.")

        storageClient = getSharedGCSStorageClient()
        applicationStorageBucket = storage.Bucket(storageClient, "kwola-testing-run-data-" + config.applicationId)
        gcsFilePath = filePath[len(config.configurationDirectory)+1:]
        objectBlob = storage.Blob(gcsFilePath, applicationStorageBucket)
        objectBlob.upload_from_string(fileData)
    else:
        raise RuntimeError(f"Unexpected value {config['data_file_storage_method']} for configuration data_file_storage_method")


@autoretry()
def loadKwolaFileData(filePath, config, printErrorOnFailure=True):
    try:
        if config['data_file_storage_method'] == 'local':
            with open(filePath, 'rb') as f:
                return f.read()
        elif config['data_file_storage_method'] == 'gcs':
            if 'applicationId' not in config or config.applicationId is None:
                raise RuntimeError("Can't load object from google cloud storage without an applicationId, which is used to indicate the bucket.")

            storageClient = getSharedGCSStorageClient()
            applicationStorageBucket = storage.Bucket(storageClient, "kwola-testing-run-data-" + config.applicationId)
            gcsFilePath = filePath[len(config.configurationDirectory)+1:]
            objectBlob = storage.Blob(gcsFilePath, applicationStorageBucket)
            data = objectBlob.download_as_string()
            return data
        else:
            raise RuntimeError(f"Unexpected value {config['data_file_storage_method']} for configuration data_file_storage_method")
    except FileNotFoundError:
        if printErrorOnFailure:
            getLogger().info(f"Error: Failed to load file {filePath}. File not found. Usually implies the file failed to write. "
                                  "Sometimes this occurs if you kill the process while it is running. If this occurs "
                                  "during normal operations without interruption, that would indicate a bug.")
        return
    except google.cloud.exceptions.NotFound:
        if printErrorOnFailure:
            getLogger().info(f"Error: Failed to load object {filePath}. Google cloud storage file not found. Usually implies the file failed to write. "
                                  "Sometimes this occurs if you kill the process while it is running. If this occurs "
                                  "during normal operations without interruption, that would indicate a bug.")
        return
