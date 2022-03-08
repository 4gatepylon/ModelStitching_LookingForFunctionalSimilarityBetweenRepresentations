from utils import combos
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

# This is it, we hardcode because its easy LOL
SMALLPAIRNUM2FILENAMES = {   
    1: ('resnet1111.pt', 'resnet1111.pt'),
    2: ('resnet1111.pt', 'resnet2111.pt'),
    3: ('resnet1111.pt', 'resnet1211.pt'),
    4: ('resnet1111.pt', 'resnet2211.pt'),
    5: ('resnet1111.pt', 'resnet1121.pt'),
    6: ('resnet1111.pt', 'resnet2121.pt'),
    7: ('resnet1111.pt', 'resnet1221.pt'),
    8: ('resnet1111.pt', 'resnet2221.pt'),
    9: ('resnet1111.pt', 'resnet1112.pt'),
    10: ('resnet1111.pt', 'resnet2112.pt'),
    11: ('resnet1111.pt', 'resnet1212.pt'),
    12: ('resnet1111.pt', 'resnet2212.pt'),
    13: ('resnet1111.pt', 'resnet1122.pt'),
    14: ('resnet1111.pt', 'resnet2122.pt'),
    15: ('resnet1111.pt', 'resnet1222.pt'),
    16: ('resnet1111.pt', 'resnet2222.pt'),
    17: ('resnet2111.pt', 'resnet1111.pt'),
    18: ('resnet2111.pt', 'resnet2111.pt'),
    19: ('resnet2111.pt', 'resnet1211.pt'),
    20: ('resnet2111.pt', 'resnet2211.pt'),
    21: ('resnet2111.pt', 'resnet1121.pt'),
    22: ('resnet2111.pt', 'resnet2121.pt'),
    23: ('resnet2111.pt', 'resnet1221.pt'),
    24: ('resnet2111.pt', 'resnet2221.pt'),
    25: ('resnet2111.pt', 'resnet1112.pt'),
    26: ('resnet2111.pt', 'resnet2112.pt'),
    27: ('resnet2111.pt', 'resnet1212.pt'),
    28: ('resnet2111.pt', 'resnet2212.pt'),
    29: ('resnet2111.pt', 'resnet1122.pt'),
    30: ('resnet2111.pt', 'resnet2122.pt'),
    31: ('resnet2111.pt', 'resnet1222.pt'),
    32: ('resnet2111.pt', 'resnet2222.pt'),
    33: ('resnet1211.pt', 'resnet1111.pt'),
    34: ('resnet1211.pt', 'resnet2111.pt'),
    35: ('resnet1211.pt', 'resnet1211.pt'),
    36: ('resnet1211.pt', 'resnet2211.pt'),
    37: ('resnet1211.pt', 'resnet1121.pt'),
    38: ('resnet1211.pt', 'resnet2121.pt'),
    39: ('resnet1211.pt', 'resnet1221.pt'),
    40: ('resnet1211.pt', 'resnet2221.pt'),
    41: ('resnet1211.pt', 'resnet1112.pt'),
    42: ('resnet1211.pt', 'resnet2112.pt'),
    43: ('resnet1211.pt', 'resnet1212.pt'),
    44: ('resnet1211.pt', 'resnet2212.pt'),
    45: ('resnet1211.pt', 'resnet1122.pt'),
    46: ('resnet1211.pt', 'resnet2122.pt'),
    47: ('resnet1211.pt', 'resnet1222.pt'),
    48: ('resnet1211.pt', 'resnet2222.pt'),
    49: ('resnet2211.pt', 'resnet1111.pt'),
    50: ('resnet2211.pt', 'resnet2111.pt'),
    51: ('resnet2211.pt', 'resnet1211.pt'),
    52: ('resnet2211.pt', 'resnet2211.pt'),
    53: ('resnet2211.pt', 'resnet1121.pt'),
    54: ('resnet2211.pt', 'resnet2121.pt'),
    55: ('resnet2211.pt', 'resnet1221.pt'),
    56: ('resnet2211.pt', 'resnet2221.pt'),
    57: ('resnet2211.pt', 'resnet1112.pt'),
    58: ('resnet2211.pt', 'resnet2112.pt'),
    59: ('resnet2211.pt', 'resnet1212.pt'),
    60: ('resnet2211.pt', 'resnet2212.pt'),
    61: ('resnet2211.pt', 'resnet1122.pt'),
    62: ('resnet2211.pt', 'resnet2122.pt'),
    63: ('resnet2211.pt', 'resnet1222.pt'),
    64: ('resnet2211.pt', 'resnet2222.pt'),
    65: ('resnet1121.pt', 'resnet1111.pt'),
    66: ('resnet1121.pt', 'resnet2111.pt'),
    67: ('resnet1121.pt', 'resnet1211.pt'),
    68: ('resnet1121.pt', 'resnet2211.pt'),
    69: ('resnet1121.pt', 'resnet1121.pt'),
    70: ('resnet1121.pt', 'resnet2121.pt'),
    71: ('resnet1121.pt', 'resnet1221.pt'),
    72: ('resnet1121.pt', 'resnet2221.pt'),
    73: ('resnet1121.pt', 'resnet1112.pt'),
    74: ('resnet1121.pt', 'resnet2112.pt'),
    75: ('resnet1121.pt', 'resnet1212.pt'),
    76: ('resnet1121.pt', 'resnet2212.pt'),
    77: ('resnet1121.pt', 'resnet1122.pt'),
    78: ('resnet1121.pt', 'resnet2122.pt'),
    79: ('resnet1121.pt', 'resnet1222.pt'),
    80: ('resnet1121.pt', 'resnet2222.pt'),
    81: ('resnet2121.pt', 'resnet1111.pt'),
    82: ('resnet2121.pt', 'resnet2111.pt'),
    83: ('resnet2121.pt', 'resnet1211.pt'),
    84: ('resnet2121.pt', 'resnet2211.pt'),
    85: ('resnet2121.pt', 'resnet1121.pt'),
    86: ('resnet2121.pt', 'resnet2121.pt'),
    87: ('resnet2121.pt', 'resnet1221.pt'),
    88: ('resnet2121.pt', 'resnet2221.pt'),
    89: ('resnet2121.pt', 'resnet1112.pt'),
    90: ('resnet2121.pt', 'resnet2112.pt'),
    91: ('resnet2121.pt', 'resnet1212.pt'),
    92: ('resnet2121.pt', 'resnet2212.pt'),
    93: ('resnet2121.pt', 'resnet1122.pt'),
    94: ('resnet2121.pt', 'resnet2122.pt'),
    95: ('resnet2121.pt', 'resnet1222.pt'),
    96: ('resnet2121.pt', 'resnet2222.pt'),
    97: ('resnet1221.pt', 'resnet1111.pt'),
    98: ('resnet1221.pt', 'resnet2111.pt'),
    99: ('resnet1221.pt', 'resnet1211.pt'),
    100: ('resnet1221.pt', 'resnet2211.pt'),
    101: ('resnet1221.pt', 'resnet1121.pt'),
    102: ('resnet1221.pt', 'resnet2121.pt'),
    103: ('resnet1221.pt', 'resnet1221.pt'),
    104: ('resnet1221.pt', 'resnet2221.pt'),
    105: ('resnet1221.pt', 'resnet1112.pt'),
    106: ('resnet1221.pt', 'resnet2112.pt'),
    107: ('resnet1221.pt', 'resnet1212.pt'),
    108: ('resnet1221.pt', 'resnet2212.pt'),
    109: ('resnet1221.pt', 'resnet1122.pt'),
    110: ('resnet1221.pt', 'resnet2122.pt'),
    111: ('resnet1221.pt', 'resnet1222.pt'),
    112: ('resnet1221.pt', 'resnet2222.pt'),
    113: ('resnet2221.pt', 'resnet1111.pt'),
    114: ('resnet2221.pt', 'resnet2111.pt'),
    115: ('resnet2221.pt', 'resnet1211.pt'),
    116: ('resnet2221.pt', 'resnet2211.pt'),
    117: ('resnet2221.pt', 'resnet1121.pt'),
    118: ('resnet2221.pt', 'resnet2121.pt'),
    119: ('resnet2221.pt', 'resnet1221.pt'),
    120: ('resnet2221.pt', 'resnet2221.pt'),
    121: ('resnet2221.pt', 'resnet1112.pt'),
    122: ('resnet2221.pt', 'resnet2112.pt'),
    123: ('resnet2221.pt', 'resnet1212.pt'),
    124: ('resnet2221.pt', 'resnet2212.pt'),
    125: ('resnet2221.pt', 'resnet1122.pt'),
    126: ('resnet2221.pt', 'resnet2122.pt'),
    127: ('resnet2221.pt', 'resnet1222.pt'),
    128: ('resnet2221.pt', 'resnet2222.pt'),
    129: ('resnet1112.pt', 'resnet1111.pt'),
    130: ('resnet1112.pt', 'resnet2111.pt'),
    131: ('resnet1112.pt', 'resnet1211.pt'),
    132: ('resnet1112.pt', 'resnet2211.pt'),
    133: ('resnet1112.pt', 'resnet1121.pt'),
    134: ('resnet1112.pt', 'resnet2121.pt'),
    135: ('resnet1112.pt', 'resnet1221.pt'),
    136: ('resnet1112.pt', 'resnet2221.pt'),
    137: ('resnet1112.pt', 'resnet1112.pt'),
    138: ('resnet1112.pt', 'resnet2112.pt'),
    139: ('resnet1112.pt', 'resnet1212.pt'),
    140: ('resnet1112.pt', 'resnet2212.pt'),
    141: ('resnet1112.pt', 'resnet1122.pt'),
    142: ('resnet1112.pt', 'resnet2122.pt'),
    143: ('resnet1112.pt', 'resnet1222.pt'),
    144: ('resnet1112.pt', 'resnet2222.pt'),
    145: ('resnet2112.pt', 'resnet1111.pt'),
    146: ('resnet2112.pt', 'resnet2111.pt'),
    147: ('resnet2112.pt', 'resnet1211.pt'),
    148: ('resnet2112.pt', 'resnet2211.pt'),
    149: ('resnet2112.pt', 'resnet1121.pt'),
    150: ('resnet2112.pt', 'resnet2121.pt'),
    151: ('resnet2112.pt', 'resnet1221.pt'),
    152: ('resnet2112.pt', 'resnet2221.pt'),
    153: ('resnet2112.pt', 'resnet1112.pt'),
    154: ('resnet2112.pt', 'resnet2112.pt'),
    155: ('resnet2112.pt', 'resnet1212.pt'),
    156: ('resnet2112.pt', 'resnet2212.pt'),
    157: ('resnet2112.pt', 'resnet1122.pt'),
    158: ('resnet2112.pt', 'resnet2122.pt'),
    159: ('resnet2112.pt', 'resnet1222.pt'),
    160: ('resnet2112.pt', 'resnet2222.pt'),
    161: ('resnet1212.pt', 'resnet1111.pt'),
    162: ('resnet1212.pt', 'resnet2111.pt'),
    163: ('resnet1212.pt', 'resnet1211.pt'),
    164: ('resnet1212.pt', 'resnet2211.pt'),
    165: ('resnet1212.pt', 'resnet1121.pt'),
    166: ('resnet1212.pt', 'resnet2121.pt'),
    167: ('resnet1212.pt', 'resnet1221.pt'),
    168: ('resnet1212.pt', 'resnet2221.pt'),
    169: ('resnet1212.pt', 'resnet1112.pt'),
    170: ('resnet1212.pt', 'resnet2112.pt'),
    171: ('resnet1212.pt', 'resnet1212.pt'),
    172: ('resnet1212.pt', 'resnet2212.pt'),
    173: ('resnet1212.pt', 'resnet1122.pt'),
    174: ('resnet1212.pt', 'resnet2122.pt'),
    175: ('resnet1212.pt', 'resnet1222.pt'),
    176: ('resnet1212.pt', 'resnet2222.pt'),
    177: ('resnet2212.pt', 'resnet1111.pt'),
    178: ('resnet2212.pt', 'resnet2111.pt'),
    179: ('resnet2212.pt', 'resnet1211.pt'),
    180: ('resnet2212.pt', 'resnet2211.pt'),
    181: ('resnet2212.pt', 'resnet1121.pt'),
    182: ('resnet2212.pt', 'resnet2121.pt'),
    183: ('resnet2212.pt', 'resnet1221.pt'),
    184: ('resnet2212.pt', 'resnet2221.pt'),
    185: ('resnet2212.pt', 'resnet1112.pt'),
    186: ('resnet2212.pt', 'resnet2112.pt'),
    187: ('resnet2212.pt', 'resnet1212.pt'),
    188: ('resnet2212.pt', 'resnet2212.pt'),
    189: ('resnet2212.pt', 'resnet1122.pt'),
    190: ('resnet2212.pt', 'resnet2122.pt'),
    191: ('resnet2212.pt', 'resnet1222.pt'),
    192: ('resnet2212.pt', 'resnet2222.pt'),
    193: ('resnet1122.pt', 'resnet1111.pt'),
    194: ('resnet1122.pt', 'resnet2111.pt'),
    195: ('resnet1122.pt', 'resnet1211.pt'),
    196: ('resnet1122.pt', 'resnet2211.pt'),
    197: ('resnet1122.pt', 'resnet1121.pt'),
    198: ('resnet1122.pt', 'resnet2121.pt'),
    199: ('resnet1122.pt', 'resnet1221.pt'),
    200: ('resnet1122.pt', 'resnet2221.pt'),
    201: ('resnet1122.pt', 'resnet1112.pt'),
    202: ('resnet1122.pt', 'resnet2112.pt'),
    203: ('resnet1122.pt', 'resnet1212.pt'),
    204: ('resnet1122.pt', 'resnet2212.pt'),
    205: ('resnet1122.pt', 'resnet1122.pt'),
    206: ('resnet1122.pt', 'resnet2122.pt'),
    207: ('resnet1122.pt', 'resnet1222.pt'),
    208: ('resnet1122.pt', 'resnet2222.pt'),
    209: ('resnet2122.pt', 'resnet1111.pt'),
    210: ('resnet2122.pt', 'resnet2111.pt'),
    211: ('resnet2122.pt', 'resnet1211.pt'),
    212: ('resnet2122.pt', 'resnet2211.pt'),
    213: ('resnet2122.pt', 'resnet1121.pt'),
    214: ('resnet2122.pt', 'resnet2121.pt'),
    215: ('resnet2122.pt', 'resnet1221.pt'),
    216: ('resnet2122.pt', 'resnet2221.pt'),
    217: ('resnet2122.pt', 'resnet1112.pt'),
    218: ('resnet2122.pt', 'resnet2112.pt'),
    219: ('resnet2122.pt', 'resnet1212.pt'),
    220: ('resnet2122.pt', 'resnet2212.pt'),
    221: ('resnet2122.pt', 'resnet1122.pt'),
    222: ('resnet2122.pt', 'resnet2122.pt'),
    223: ('resnet2122.pt', 'resnet1222.pt'),
    224: ('resnet2122.pt', 'resnet2222.pt'),
    225: ('resnet1222.pt', 'resnet1111.pt'),
    226: ('resnet1222.pt', 'resnet2111.pt'),
    227: ('resnet1222.pt', 'resnet1211.pt'),
    228: ('resnet1222.pt', 'resnet2211.pt'),
    229: ('resnet1222.pt', 'resnet1121.pt'),
    230: ('resnet1222.pt', 'resnet2121.pt'),
    231: ('resnet1222.pt', 'resnet1221.pt'),
    232: ('resnet1222.pt', 'resnet2221.pt'),
    233: ('resnet1222.pt', 'resnet1112.pt'),
    234: ('resnet1222.pt', 'resnet2112.pt'),
    235: ('resnet1222.pt', 'resnet1212.pt'),
    236: ('resnet1222.pt', 'resnet2212.pt'),
    237: ('resnet1222.pt', 'resnet1122.pt'),
    238: ('resnet1222.pt', 'resnet2122.pt'),
    239: ('resnet1222.pt', 'resnet1222.pt'),
    240: ('resnet1222.pt', 'resnet2222.pt'),
    241: ('resnet2222.pt', 'resnet1111.pt'),
    242: ('resnet2222.pt', 'resnet2111.pt'),
    243: ('resnet2222.pt', 'resnet1211.pt'),
    244: ('resnet2222.pt', 'resnet2211.pt'),
    245: ('resnet2222.pt', 'resnet1121.pt'),
    246: ('resnet2222.pt', 'resnet2121.pt'),
    247: ('resnet2222.pt', 'resnet1221.pt'),
    248: ('resnet2222.pt', 'resnet2221.pt'),
    249: ('resnet2222.pt', 'resnet1112.pt'),
    250: ('resnet2222.pt', 'resnet2112.pt'),
    251: ('resnet2222.pt', 'resnet1212.pt'),
    252: ('resnet2222.pt', 'resnet2212.pt'),
    253: ('resnet2222.pt', 'resnet1122.pt'),
    254: ('resnet2222.pt', 'resnet2122.pt'),
    255: ('resnet2222.pt', 'resnet1222.pt'),
    256: ('resnet2222.pt', 'resnet2222.pt'),
}

if __name__ == "__main__":
    combinations = combos(4, [1, 2])
    assert len(combinations) == 16
    mapping = {}
    num = 1
    for i in range(16):
        for j in range(16):
            mapping[num] = (
                "resnet"+"".join(map(lambda c: str(c), combinations[i]))+".pt",
                "resnet"+"".join(map(lambda c: str(c), combinations[j]))+".pt"
            )
            num += 1
    pp.pprint(mapping)
