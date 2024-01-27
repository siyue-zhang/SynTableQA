import re

def lower(s):
    # Convert everything except text between (single or double) quotation marks to lower case
    return re.sub(
        r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s
    )

a = 'SELECT Official_native_language FROM country WHERE Official_native_language LIKE "%English%"'
print(lower(a))