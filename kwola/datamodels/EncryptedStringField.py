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
