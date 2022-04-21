from argparse import ArgumentParser

from typing import Dict, List, Tuple

# Pretty printer is invaluable for debuggings
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

TEST_LOG_FILE = 'r1111-r1111.log'


class Label(object):
    def __init__(self, label_str):
        self.label_str = label_str

    def isBlock(self):
        return self.label_str[0] == "(" and self.label_str[-1] == ")"

    def get(self):
        if self.isBlock():
            # Does NOT support spaces
            contents = self.label_str[1:-1]
            n1_str, n2_str = contents.split(",")
            return (int(n1_str), int(n2_str))

    def __repr__(self) -> str:
        return self.label_str

    def __str__(self) -> str:
        return self.label_str


def parse_percent_or_float(s):
    if s[-1] == '%':
        return float(s[:-1]) / 100.0
    else:
        return float(s)


def parse_int(string, idx):
    val = int(string[idx])
    idx += 1
    while idx < len(string) and string[idx].isdigit():
        val = val * 10 + int(string[idx])
        idx += 1
    return val, idx


def parse_entry(string, idx):
    # NOTE: only single spaces are supported at any given place in the log
    # Parse the index entry
    assert string[idx] == "(", f"Expected [{idx}]='(' got {string[idx]}"
    idx += 1
    index1, next_i = parse_int(string, idx)
    idx = next_i
    assert string[idx] == ",", f"Expected [{idx}]=',' got {string[idx]}"
    idx += 1
    if string[idx] == " ":
        idx += 1
    assert string[idx].isdigit(), f"Expected [{idx}]=digit got {string[idx]}"
    index2, next_i = parse_int(string, idx)
    idx = next_i
    assert string[idx] == ")", f"Expected [{idx}]=')' got {string[idx]}"
    idx += 1
    # Go to parse the label entry
    assert string[idx] == ":", f"Expected [{idx}]=':' got {string[idx]}"
    idx += 1
    if string[idx] == " ":
        idx += 1
    assert string[idx] == "(", f"Expected [{idx}]='(' got {string[idx]}"
    idx += 1
    # conv or fc or (x, y)

    labels = [None, None]
    for i in range(len(labels)):
        # make sure that we start at either conv1, fc, or (x, y)
        assert string[idx] == "(" or string[idx] == "f" or string[idx] == "c", "Expected (x, y), fc, or conv1"
        if string[idx] == "f":
            # parse "fc"
            labels[i] = Label("fc")
            idx += 2
        elif string[idx] == "c":
            # parse "conv1"
            labels[i] = Label("conv1")
            idx += 5
        else:
            # parse the (x, y) tuple
            idx += 1
            val1, next_i = parse_int(string, idx)
            idx = next_i
            assert string[idx] == ",", f"Expected [{idx}]=',' got {string[idx]}"
            idx += 1
            if string[idx] == " ":
                idx += 1
            val2, next_i = parse_int(string, idx)
            idx = next_i
            assert string[idx] == ")", f"Expected [{idx}]=')' got {string[idx]}"
            idx += 1
            labels[i] = Label(f"({val1},{val2})")
        # make sure to skip over the comma and space
        if i == 0:
            assert string[idx] == ",", f"Expected [{idx}]=',' got {string[idx]}"
            idx += 1
            if string[idx] == " ":
                idx += 1
    # sanity check that the label pair closes
    assert string[idx] == ")", f"Expected [{idx}]=')' got {string[idx]}"
    idx += 1
    # check that there is a new label
    if idx < len(string) - 1:
        assert string[idx] == ",", f"Expected [{idx}]=',' got {string[idx]}"
        idx += 1
        assert string[idx] == "(", "failed to open new block"
    assert not labels[0] is None and not labels[
        1] is None, f"failed to set labels: {labels[0]}, {labels[1]}"
    return idx, ((index1, index2), (labels[0], labels[1]))


class StitchInfo(object):
    NUM_LINES = 6

    def __init__(self, net1, net2, label1, label2):
        # These were hardcoded in the experiment we did way back when
        self.num_epochs_vanilla_stitch = 4
        self.num_epochs_autoencoder_stitch = 30

        self.net1 = net1
        self.net2 = net2

        self.label1 = label1
        self.label2 = label2

        self.vanilla_acc = None
        self.autoencoder_acc = None

        # Mean2 errors
        self.vanilla_orig_mean2 = None
        self.autoencoder_orig_mean2 = None
        self.vanilla_autoencoder_mean2 = None

    @ staticmethod
    def from_lines(lines, net1, net2, label1, label2):
        # Example:
        # original accuracies were 0.9347345132743363 and 0.9347345132743363 <- NOTE these are NOT percentages
        # accuracy of vanilla stitch model is 0.9142699115044249 <- NOTE this is NOT a percentage
        # vanilla stitch mean2 difference is 0.019725045065085094
        # accuracy of autoencoder stitch model is 25.110619469026545 <- NOTE this is a percentage
        # autoencoder stitch mean2 difference is 0.00837538814984071
        # vanilla stitch autoencoder mean2 difference is 0.005824266808728377
        assert len(lines) == StitchInfo.NUM_LINES
        assert lines[0].startswith(
            "original accuracies were")
        assert lines[1].startswith(
            "accuracy of vanilla stitch model")
        assert lines[2].startswith(
            "vanilla stitch mean2 difference")
        assert lines[3].startswith(
            "accuracy of autoencoder stitch model")
        assert lines[4].startswith(
            "autoencoder stitch mean2 difference")
        assert lines[5].startswith(
            "vanilla stitch autoencoder mean2 difference")

        # Might want to assert that first line has the right accuracies based
        # on the ExperimentInfo which contains this
        stitch_info = StitchInfo(net1, net2, label1, label2)
        stitch_info.vanilla_acc = float(lines[1].split(" ")[-1])
        # For some reason this is the only one that was a percentage
        stitch_info.autoencoder_acc = float(lines[3].split(" ")[-1]) / 100.0
        stitch_info.vanilla_orig_mean2 = float(lines[2].split(" ")[-1])
        stitch_info.autoencoder_orig_mean2 = float(lines[4].split(" ")[-1])
        stitch_info.vanilla_autoencoder_mean2 = float(lines[5].split(" ")[-1])
        return stitch_info


class ExperimentInfo(object):
    def __init__(self, logfile):
        self.name1: str = None
        self.name2: str = None

        self.numbers1: List[int] = None
        self.numbers2: List[int] = None

        self.acc1: float = None
        self.acc1_rand: float = None
        self.acc2: float = None
        self.acc2_rand: float = None

        self.idx2labels: Dict[Tuple[int, int], Tuple[Label, Label]] = None
        self.labels2stitchinfo_orig_orig: Dict[Tuple[Label,
                                                     Label], StitchInfo] = None
        self.labels2stitchinfo_orig_rand: Dict[Tuple[Label,
                                                     Label], StitchInfo] = None
        self.labels2stitchinfo_rand_orig: Dict[Tuple[Label,
                                                     Label], StitchInfo] = None
        self.labels2stitchinfo_rand_rand: Dict[Tuple[Label,
                                                     Label], StitchInfo] = None

        with open(args.input, "r") as f:
            contents = f.read()
        self.parse(contents)

        assert not self.name1 is None
        assert not self.name2 is None
        assert not self.numbers1 is None
        assert not self.numbers2 is None

        assert not self.acc1 is None
        assert not self.acc1_rand is None
        assert not self.acc2 is None
        assert not self.acc2_rand is None

        # Make sure the parser was able to extract this from the header
        assert not self.idx2labels is None
        # TODO
        # Assert that the parser was able to find each of these
        # assert not self.labels2stitchinfo_rand_rand is None
        # assert not self.labels2stitchinfo_orig_rand is None
        # assert not self.labels2stitchinfo_rand_orig is None
        # assert not self.labels2stitchinfo_orig_orig is None
        # Sanity test that we got the same sizes for each of these (it's a table in the form of a dictionary)
        # assert len(self.labels2stitchinfo_orig_orig) == len(self.labels2stitchinfo_orig_rand)
        # assert len(self.labels2stitchinfo_orig_orig) == len(self.labels2stitchinfo_rand_orig)
        # assert len(self.labels2stitchinfo_orig_orig) == len(self.labels2stitchinfo_rand_rand)
        # assert len(self.labels2stitchinfo_orig_orig) == len(self.idx2labels)

    def parse_global_header(self, header):
        # Example:
        # 0:                 importing
        # 1:                 done importing
        # 2:                 will stitch resnet_1111.pt and resnet_1111.pt in ../pretrained_resnets
        # 3:                 numbers1: [1, 1, 1, 1]
        # 4:                 numbers2: [1, 1, 1, 1]
        # 5:                 name1: resnet_1111
        # 6:                 name2: resnet_1111
        # 7:                 loading models
        # 8:                 getting loaders
        # 9:                 evaluating accuracies of pretrained models
        # 10:                accuracy of resnet_1111 is 93.47345132743364%
        # 11:                accuracy of resnet_1111 random is 10.564159292035399%
        # 12:                accuracy of resnet_1111 is 93.47345132743364%
        # 13:                accuracy of resnet_1111 random is 10.398230088495575%
        # 14:                initializing the stitches (note all idx2label are the same)
        # 15:                {   (0, 1): ('conv1', (1, 0)),
        # 16:                (0, 2): ('conv1', (2, 0)),
        # 17:                (0, 3): ('conv1', (3, 0)),
        # 18:                (0, 4): ('conv1', (4, 0)),
        # 19:                (0, 5): ('conv1', 'fc'),
        # 20:                (1, 1): ((1, 0), (1, 0)),
        # 21:                (1, 2): ((1, 0), (2, 0)),
        # 22:                (1, 3): ((1, 0), (3, 0)),
        # 23:                (1, 4): ((1, 0), (4, 0)),
        # 24:                (1, 5): ((1, 0), 'fc'),
        # 25:                (2, 1): ((2, 0), (1, 0)),
        # 26:                (2, 2): ((2, 0), (2, 0)),
        # 27:                (2, 3): ((2, 0), (3, 0)),
        # 28:                (2, 4): ((2, 0), (4, 0)),
        # 29:                (2, 5): ((2, 0), 'fc'),
        # 30:                (3, 1): ((3, 0), (1, 0)),
        # 31:                (3, 2): ((3, 0), (2, 0)),
        # 32:                (3, 3): ((3, 0), (3, 0)),
        # 33:                (3, 4): ((3, 0), (4, 0)),
        # 34:                (3, 5): ((3, 0), 'fc'),
        # 35:                (4, 1): ((4, 0), (1, 0)),
        # 36:                (4, 2): ((4, 0), (2, 0)),
        # 37:                (4, 3): ((4, 0), (3, 0)),
        # 38:                (4, 4): ((4, 0), (4, 0)),
        # 39:                (4, 5): ((4, 0), 'fc')}
        # 40:                stitching, will save in sims_resnet_1111_resnet_1111

        # Sanity test that the header is what we expect because otherwise there may be a bug in
        # this parser or we may be parsing the wrong logfiles
        assert header[0] == "importing"
        assert header[1] == "done importing"
        assert "will stitch" in header[2] and "in ../pretrained_resnets" in header[2]
        assert header[3].startswith("numbers1:")
        assert header[4].startswith("numbers2:")
        assert header[5].startswith("name1:")
        assert header[6].startswith("name2:")
        assert header[7] == "loading models"
        assert header[8] == "getting loaders"
        assert header[9] == "evaluating accuracies of pretrained models"
        assert header[10].startswith("accuracy of") and\
            not "rand" in header[10]
        assert header[11].startswith("accuracy of") and\
            "rand" in header[11]
        assert header[12].startswith("accuracy of") and not \
            "rand" in header[12]
        assert header[13].startswith("accuracy of") and \
            "rand" in header[13]
        assert header[14] == "initializing the stitches (note all idx2label are the same)"
        assert header[15].startswith("{")
        assert header[-2].endswith("}")
        assert "stitching, will save in" in header[-1]

        # Get the numbers and remove the brackets [ and ]
        numbers1_str = header[3].split(":")[1].strip()
        numbers2_str = header[4].split(":")[1].strip()
        numbers1_str = numbers1_str[1:-1]
        numbers2_str = numbers2_str[1:-1]
        self.numbers1 = [
            int(x) for x in map(str.strip, numbers1_str.split(","))
        ]
        self.numbers2 = [
            int(x) for x in map(str.strip, numbers2_str.split(","))
        ]
        # Get the name
        self.name1 = header[5].split(":")[1].strip()
        self.name2 = header[6].split(":")[1].strip()

        # Sanity test that the logfile is consistent, since otherwise there may be a bug in this parser
        _tmp1 = "".join(map(str, self.numbers1))
        _tmp2 = "".join(map(str, self.numbers2))
        assert self.name1 == f"resnet_{_tmp1}"
        assert self.name2 == f"resnet_{_tmp2}"
        assert self.name1 in header[10]
        assert self.name1 in header[11]
        assert self.name2 in header[12]
        assert self.name2 in header[13]

        # Get the accuracies (always last number of the line if you look at the example)
        acc1_str = header[10].split(" ")[-1].strip()
        acc1_rand_str = header[11].split(" ")[-1].strip()
        acc2_str = header[12].split(" ")[-1].strip()
        acc2_rand_str = header[13].split(" ")[-1].strip()

        self.acc1 = parse_percent_or_float(acc1_str)
        self.acc1_rand = parse_percent_or_float(acc1_rand_str)
        self.acc2 = parse_percent_or_float(acc2_str)
        self.acc2_rand = parse_percent_or_float(acc2_rand_str)

        # Sanity test that this code is working properly
        assert 1.0 > self.acc1 and self.acc1 > 0.0
        assert 1.0 > self.acc1_rand and self.acc1_rand > 0.0
        assert 1.0 > self.acc2 and self.acc2 > 0.0
        assert 1.0 > self.acc2_rand and self.acc2_rand > 0.0

        # We can't really use json.parse since it doesn't support tuples; anyways, this is OK because it's not nested
        # Get the string for the idx2labels
        idx2labels_str = "".join(header[15:-1])
        # We remove the { and } and then have to split by ), because , would catch 1, 0 inside the parens
        idx2labels_str = idx2labels_str[1:-1]
        idx2labels_str = "".join(
            list(filter(lambda x: x != "'", idx2labels_str)),
        )
        idx2labels_str = idx2labels_str.strip()
        entries = []
        i = 0
        while i < len(idx2labels_str):
            next_i, entry = parse_entry(idx2labels_str, i)
            entries.append(entry)
            i = next_i

        self.idx2labels = {idxs: labels for idxs, labels in entries}
        # pp.pprint(self.idx2labels)

    # TODO
    def parse_blocks_with_local_headers(self, blocks):
        # raise NotImplementedError
        pass

    def parse(self, contents):
        # This delimiter is missing on the very first block unfortunately but otherwise works
        block_strs = contents.split("***")

        # Remove tabs and put everything in lower case
        blocks = [block.split("\n") for block in block_strs]
        blocks = [[line.strip().lower() for line in block] for block in blocks]

        # Remove those pesky "epoch" lines and "saving" lines that tell you what file
        # some tensor of similarities (or whatever) was saved to (not important yet)
        blocks = [list(filter(lambda line: not "epoch" in line and len(line) > 0 and not "saving" in line, lines))
                  for lines in blocks]
        blocks = list(filter(lambda block: len(block) > 0, blocks))

        # remove the intro block
        intro_block = blocks[0]
        first_block = intro_block[-8:]
        blocks[0] = first_block

        # global header tells you which network architectures are being stitched, etcetera
        global_header = intro_block[:-8]
        self.parse_global_header(global_header)

        # clean = "\n\n".join(map(lambda lines: "\n".join(
        #     map(lambda line: f"\t{line}", lines)), blocks))
        # print(clean)

        # local header tells you which networks are being stitched
        self.parse_blocks_with_local_headers(blocks)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=TEST_LOG_FILE,
        required=False,
    )
    args = parser.parse_args()
    info = ExperimentInfo(args.input)
