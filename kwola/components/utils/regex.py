import re

sharedUrlRegex = re.compile(
    r'(?:http|ftp|ws)s?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:[/?]\S+|/?)', re.IGNORECASE)

sharedNonJavascriptCodeUrlRegex = re.compile(
    r'(?:http|ftp|ws)s?://'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?!\S+\.js)'  # Don't match anything that is .js
    r'(?:[/?]\S+|/?)', re.IGNORECASE)


sharedHexUuidRegex = re.compile(
    r'[a-f0-9]{8}-?[a-f0-9]{4}-?[a-f0-9]{4}-?[a-f0-9]{4}(-?[a-f0-9]{12})?(-?[a-f0-9]{8})?',
    re.IGNORECASE)


sharedMongoObjectIdRegex = re.compile(
    r'(5(?=\d{0,22}[a-f])[a-f0-9]{23})|(5(?=\d{0,14}[a-f])[a-f0-9]{15})',
    re.IGNORECASE)


sharedISO8601DateRegex = re.compile(
    r'20\d{2}-\d\d-\d\d(?:T\d\d:\d\d:\d\d(?:\.\d{6})?(?:[+-]\d\d(?::\d\d)?)?)?')


sharedISO8601TimeRegex = re.compile(
    r'\d\d:\d\d:\d\d(?:[.:]\d{1,6})?')


sharedStandardBase64Regex = re.compile(
    r'((?=[a-zA-Z+/]*?\d)(?=[0-9a-zA-Z]*?[+/])(?=[0-9A-Z+/]*?[a-z])(?=[0-9a-z+/]*?[A-Z])[a-zA-Z0-9+/]{16,}={1,2})'
    r'|'
    r'((?=[a-zA-Z+/]*?\d)(?=[0-9a-zA-Z+]*?[/])(?=[0-9a-zA-Z/]*?[+])(?=[0-9A-Z+/]*?[a-z])(?=[0-9a-z+/]*?[A-Z])[a-zA-Z0-9+/]{16,}={0,2})'
    r'|'
    r'((?=[a-zA-Z-_]*?\d)(?=[0-9a-zA-Z]*?[-_])(?=[0-9A-Z-_]*?[a-z])(?=[0-9a-z-_]*?[A-Z])[a-zA-Z0-9-_]{16,}(=|%3[dD]){1,2})'
    r'|'
    r'((?=[a-zA-Z-_]*?\d)(?=[0-9a-zA-Z-]*?[_])(?=[0-9a-zA-Z_]*?[-])(?=[0-9A-Z-_]*?[a-z])(?=[0-9a-z-_]*?[A-Z])[a-zA-Z0-9-_]{16,}(=|%3[dD]){0,2})')


sharedAlphaNumericalCodeRegex = re.compile(
    r'(?=[a-zA-Z]*\d)(?=[0-9]*[a-zA-Z])[a-zA-Z0-9]{16,}')


sharedIPAddressRegex = re.compile(
    r'\D\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?=\D)')


sharedLongNumberRegex = re.compile(
    r'\d{8,}')


