#
#     This file is copyright 2020 Kwola Software Testing Inc.
#     All Rights Reserved.
#


from mongoengine import *
from faker import Faker
import random
import math

globalFakeStringGenerator = Faker()

class TypingActionConfiguration(EmbeddedDocument):
    type = StringField()

    biasKeywords = StringField()

    text = StringField()

    generateRandom = BooleanField()

    includeCity = BooleanField()
    includeCountry = BooleanField()
    includePostalCode = BooleanField()
    canadianStylePostalCode = BooleanField()
    ukStylePostalCode = BooleanField()
    sixDigitZipCode = BooleanField()
    fiveDigitZipCode = BooleanField()
    fourDigitZipCode = BooleanField()
    nineDigitZipCode = BooleanField()

    includeExtension = BooleanField()
    includeCountryCode = BooleanField()

    dateFormat = StringField()
    timeFormat = StringField()

    numberMinimumValue = FloatField()
    numberMaximumValue = FloatField()
    includeDecimals = BooleanField()
    distributeLogarithmically = BooleanField()

    minimumLength = IntField()
    maximumLength = IntField()
    characterSet = StringField()

    def randomString(self, chars, len):
        """
            Generates a random string.

            Just a utility function.

            :param chars: A string containing possible characters to select from.
            :param len: The number of characters to put into the generated string.
            :return:
        """
        base = ""
        for n in range(len):
            base += str(random.choice(chars))
        return base

    def generateText(self):
        if not self.generateRandom:
            return self.text

        if self.type == "email":
            return "testing_" + self.randomString('0123456789', 10) + "@kwola.io"

        if self.type == "address":
            text = globalFakeStringGenerator.street_address()

            if self.includeCity:
                text += ", " + globalFakeStringGenerator.city()

            if self.includeCountry:
                text += ", " + globalFakeStringGenerator.country()

            if self.includePostalCode:
                text += ", " + globalFakeStringGenerator.postcode()

            return text

        if self.type == "city":
            return globalFakeStringGenerator.city()

        if self.type == "country":
            return globalFakeStringGenerator.country()

        if self.type == "postalcode":
            allowedTypes = []
            if self.canadianStylePostalCode:
                allowedTypes.append("canadianStylePostalCode")
            if self.ukStylePostalCode:
                allowedTypes.append("ukStylePostalCode")
            if self.canadianStylePostalCode:
                allowedTypes.append("canadianStylePostalCode")
            if self.sixDigitZipCode:
                allowedTypes.append("sixDigitZipCode")
            if self.fiveDigitZipCode:
                allowedTypes.append("fiveDigitZipCode")
            if self.fourDigitZipCode:
                allowedTypes.append("fourDigitZipCode")
            if self.nineDigitZipCode:
                allowedTypes.append("nineDigitZipCode")

            chosen = random.choice(allowedTypes)
            if chosen == "canadianStylePostalCode":
                letter1 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                letter2 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                letter3 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                digit1 = random.choice('0123456789')
                digit2 = random.choice('0123456789')
                digit3 = random.choice('0123456789')

                return letter1 + digit1 + letter2 + digit2 + letter3 + digit3

            if chosen == "ukStylePostalCode":
                # This needs to be improved because not all the postal codes generated from this code are actually valid
                letter1 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                letter2 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                letter3 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                letter4 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                letter5 = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

                digit1 = random.choice('0123456789')
                digit2 = random.choice('0123456789')
                digit3 = random.choice('0123456789')

                format = random.choice([1,2,3,4,5])
                if format == 1:
                    return letter1 + letter2 + digit1 + letter3 + " " + digit2 + letter4 + letter5
                elif format == 2:
                    return letter1 + digit1 + letter2 + " " + digit2 + letter3 + letter4
                if format == 3:
                    return letter1 + digit1 + " " + digit2 + letter2 + letter3
                if format == 4:
                    return letter1 + digit1 + digit2 + " " + digit3 + letter2 + letter3
                if format == 5:
                    return letter1 + letter2 + digit1 + " " + digit2 + letter3 + letter4
                if format == 6:
                    return letter1 + letter2 + digit1 + digit2 + " " + digit3 + letter3 + letter4

            if chosen == "sixDigitZipCode":
                return self.randomString('0123456789', 6)
            if chosen == "fiveDigitZipCode":
                return self.randomString('0123456789', 5)
            if chosen == "fourDigitZipCode":
                return self.randomString('0123456789', 4)
            if chosen == "nineDigitZipCode":
                return self.randomString('0123456789', 9)

        if self.type == "url":
            return "http://testing-" + self.randomString('0123456789', 4) + ".kwola.io/"

        if self.type == "creditcard":
            return globalFakeStringGenerator.credit_card_number()

        if self.type == "creditcardcvc":
            return self.randomString('123456789', 3)

        if self.type == "phonenumber":
            if self.includeExtension and self.includeCountryCode:
                return globalFakeStringGenerator.phone_number()

            text = ""
            if self.includeCountryCode and random.random() > 0.5:
                text += "+" + self.randomString('123456789', 2) + "-"

            text += self.randomString('123456789', 3)
            text += "-" + self.randomString('123456789', 3)
            text += "-" + self.randomString('123456789', 4)

            if self.includeExtension and random.random() > 0.5:
                text += "x" + self.randomString('0123456789', random.randint(1, 4))

            return text

        if self.type == "date":
            return globalFakeStringGenerator.date(pattern=self.dateFormat)

        if self.type == "time":
            return globalFakeStringGenerator.date(pattern=self.dateFormat)

        if self.type == "number":
            if self.distributeLogarithmically:
                if self.numberMinimumValue < 0 and self.numberMaximumValue > 0:
                    negative = random.choice([True, False])
                    if negative:
                        range = math.log10(-self.numberMinimumValue)
                        value = -math.pow(10, random.random() * range)
                    else:
                        range = math.log10(self.numberMaximumValue)
                        value = math.pow(10, random.random() * range)
                else:
                    range = math.log10(self.numberMaximumValue - self.numberMinimumValue)
                    value = math.pow(10, random.random() * range) + self.numberMinimumValue
            else:
                range = self.numberMaximumValue - self.numberMinimumValue
                value = random.random() * range + self.numberMinimumValue

            if not self.includeDecimals:
                value = int(value)
            return str(value)

        if self.type == "letters":
            length = random.randint(self.minimumLength, self.maximumLength)
            return self.randomString(self.characterSet, length)

        return ""



