CHARS_TO_PREDICT = 3

# encodings for languages of characters
ENG = 0
RUS = 1
CHI = 2
UNK = 3
SPACE_DELIMITED_LANGS = {ENG, RUS}
NON_SPACE_DELIMITED_LANGS = {CHI}

# Prefix token for ngrams or anything that needs to start with nonempty context
PREFIX_TOKEN = "<sos>"

# returns the language of the character 'ch'
# (note this does not take into account edge cases, like japanese having
# the same unicode characters as chinese)
# Either: ENG, RUS, CHI, or UNK if could not be determined
def get_lang(ch):
    code = ord(ch)
    if 0x0041 <= code <= 0x007A or 0x00C0 <= code <= 0x024F:
        return ENG
    elif 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F:
        return RUS
    elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return CHI
    else:
        return UNK

# returns the list of tokens that make up a line
def tokenize_line(line):
        line_with_spaces_around_tokens = (
            "".join(
                f" {c} " if get_lang(c) in NON_SPACE_DELIMITED_LANGS else c
                for c in line
            )
        )
        return line_with_spaces_around_tokens.split()

def filter_prefix(prefix, start, strings):
    return [
        string for string in strings if (
            len(string) > start + len(prefix) and
            string[start:start + len(prefix)] == prefix
        )
    ]

SAMPLE_ENGLISH_DATA = [
    "The cat sat on the mat.",
    "A dog barked loudly outside.",
    "She opened the window slowly.",
    "He drank a cup of coffee.",
    "The sun rose over the hills.",
    "They walked home together.",
    "I forgot my keys again.",
    "The book fell off the table.",
    "Rain tapped against the glass.",
    "The child laughed happily.",
    "She wrote a short email.",
    "He closed the heavy door.",
    "The car stopped at the light.",
    "Birds sang in the morning.",
    "The room felt unusually cold.",
    "I checked the time twice.",
    "The teacher explained the rule.",
    "We waited in a long line.",
    "The phone rang suddenly.",
    "She tied her shoes tightly.",
    "The train arrived on time.",
    "He washed his hands carefully.",
    "The store opened early today.",
    "I heard a strange noise.",
    "The cake smelled delicious.",
    "She smiled at the camera.",
    "one small step.",
    "That's one small",
    "one giant leap",
    "leap for mankind"
]