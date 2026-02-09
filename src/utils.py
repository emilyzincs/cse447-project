import unicodedata
import string

CHARS_TO_PREDICT = 3

# encodings for languages of characters
ENG = 0
RUS = 1
CHI = 2
UNK = 3

# characters corresponding to each language
ENG_CHARS = string.ascii_lowercase
RUS_CHARS = 'ё' + ''.join(chr(c) for c in range(0x0430, 0x044F + 1))
CHI_CHARS = ''.join(chr(c) for c in range(0x4E00, 0x9FFF + 1))


# returns the language of the character 'ch'
# (note this does not take into account edge cases, like japanese having
# the same unicode characters as chinese)
# Either: ENG, RUS, CHI, or UNK if could not be determined
def get_lang(ch):
    code = ord(ch)
    # English / Latin letters
    if 0x0041 <= code <= 0x007A or 0x00C0 <= code <= 0x024F:
        return ENG
    # Russian / Cyrillic letters
    elif 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F:
        return RUS
    # Chinese characters
    elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return CHI
    else:
        return UNK

# returns whether the given character 'ch' is a unicode punctuation mark
def is_punctuation(ch):
    return unicodedata.category(ch).startswith("P")

# returns the last language in the string 'unprocessed_line'
# either ENG, RUS, CHI, or UNK if could not be determined
def get_last_lang(unprocessed_line):
    for i in range(len(unprocessed_line) - 1, -1, -1):
        ch = unprocessed_line[i]
        if ch.isspace() or is_punctuation(ch):
            continue
        return get_lang(ch)
    return UNK
