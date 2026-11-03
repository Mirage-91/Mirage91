EVENT = {
    "TARGET": 1,
    "NONTARGET": 2
}

TARGET_CHAR_DECODING = {
    11: "A",
    12: "B",
    13: "C",
    14: "D",
    15: "E",
    16: "F",
    21: "G",
    22: "H",
    23: "I",
    24: "J",
    25: "K",
    26: "L",
    31: "M",
    32: "N",
    33: "O",
    34: "P",
    35: "Q",
    36: "R",
    41: "S",
    42: "T",
    43: "U",
    44: "V",
    45: "W",
    46: "X",
    51: "Y",
    52: "Z",
    53: ".",
    54: "_",
    55: "del",
    56: "clr",
}

TARGET_CHAR_ENCODING = {v: k for k, v in TARGET_CHAR_DECODING.items()}

def is_target_event(event_code):
    """Event is target, if the first digit of custom event code is 1."""
    return str(event_code)[0] == str(EVENT["TARGET"])

def is_nontarget_event(event_code):
    """Event is non-target, if the first digit of custom event code is 2."""
    return str(event_code)[0] == str(EVENT["NONTARGET"])
