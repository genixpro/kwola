#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

import mongoengine
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import hashlib
import base64
import os

class EncryptedStringField(mongoengine.fields.StringField):
    encryptedStringPrefix = "encryptedbase64_icvorsEykshArnAvposhan5DrynoujAy:"

    def __init__(self, **kwargs):
        super(EncryptedStringField, self).__init__(**kwargs)

    def validate(self, value):
        return True

    def to_mongo(self, value):
        if value.startswith(EncryptedStringField.encryptedStringPrefix):
            return value
        else:
            return EncryptedStringField.encrypt(value)

    def to_python(self, value):
        if value.startswith(EncryptedStringField.encryptedStringPrefix):
            return EncryptedStringField.decrypt(value)
        else:
            return value

    @staticmethod
    def encrypt(value):
        encryptionKey = os.getenv("KWOLA_FIELD_ENCRYPTION_KEY")
        if encryptionKey:
            nonceData = os.urandom(16)

            keyHash = hashlib.sha256()
            keyHash.update(bytes(encryptionKey, "utf8"))
            cipher = Cipher(algorithms.AES(keyHash.digest()), modes.CTR(nonceData))

            encryptor = cipher.encryptor()
            encryptedData = nonceData + encryptor.update(bytes(value, 'utf8')) + encryptor.finalize()

            return EncryptedStringField.encryptedStringPrefix + str(base64.b64encode(encryptedData), 'ascii')
        else:
            return value

    @staticmethod
    def decrypt(value):
        encryptionKey = os.getenv("KWOLA_FIELD_ENCRYPTION_KEY")
        if encryptionKey:
            keyHash = hashlib.sha256()
            keyHash.update(bytes(encryptionKey, "utf8"))

            value = base64.b64decode(bytes(value[len(EncryptedStringField.encryptedStringPrefix):], 'ascii'))

            nonceData = value[:16]
            contents = value[16:]

            cipher = Cipher(algorithms.AES(keyHash.digest()), modes.CTR(nonceData))

            decryptor = cipher.decryptor()
            decryptedData = decryptor.update(contents) + decryptor.finalize()

            return str(decryptedData, 'utf8')
        else:
            return value
