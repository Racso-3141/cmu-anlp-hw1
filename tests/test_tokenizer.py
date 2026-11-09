import json
from transformers import AutoTokenizer
from pathlib import Path

from src.tokenizer.bpe import ASCIIBPETokenizer, string_to_ascii

from src.pytest_utils.decorators import max_score

# Q1.1: you write the bodies of the three test_not_* functions below. They are
# graded in your report, not by point value here -- but they must pass.


def test_not_injective():
    """Q1.1.A: two DIFFERENT strings that encode to the SAME token ids."""
    # todo: find two distinct strings whose encodings are identical
    raise NotImplementedError


def test_not_invertible():
    """Q1.1.A: a string that does NOT survive encode-then-decode."""
    # todo: find a string that does not round-trip
    raise NotImplementedError


def test_not_preserving_concat():
    """Q1.1.A: two strings where tokenize(a+b) != tokenize(a) + tokenize(b)."""
    # todo: find a pair whose concatenation merges across the boundary
    raise NotImplementedError

@max_score(3)
def test_encode():
    tokenizer = ASCIIBPETokenizer()
    random_ascii = r"[(A{u s=#M \tF\@`P({xAS%"
    assert tokenizer.encode(random_ascii) == string_to_ascii(random_ascii)

    tokenizer = ASCIIBPETokenizer.from_data("123123", 1)
    ####### after a single merge #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    assert tokenizer.encode(random_ascii) == string_to_ascii(random_ascii)
    assert tokenizer.encode("11") == [49, 49]
    assert tokenizer.encode("12") == [128]
    assert tokenizer.encode("23") == [50, 51]
    assert tokenizer.encode("123123") == [128, 51, 128, 51]
    assert tokenizer.encode("12312312") == [128, 51, 128, 51, 128]

    tokenizer = ASCIIBPETokenizer.from_data("123123", 2)
    ####### after two merges #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    # "123" -> 129
    assert tokenizer.encode(random_ascii) == string_to_ascii(random_ascii)
    assert tokenizer.encode("11") == [49, 49]
    assert tokenizer.encode("12") == [128]
    assert tokenizer.encode("23") == [50, 51]
    assert tokenizer.encode("123") == [129]
    assert tokenizer.encode("123123") == [129, 129]
    assert tokenizer.encode("12312312") == [129, 129, 128]


@max_score(3)
def test_decode():
    tokenizer = ASCIIBPETokenizer()
    assert (
        tokenizer.decode([97, 102, 101, 106, 117, 100, 111, 97, 119, 101, 114])
        == "afejudoawer"
    )


@max_score(3)
def test_one_merge():
    tokenizer = ASCIIBPETokenizer.from_data("123123", 1)
    ####### after a single merge #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    assert len(tokenizer.vocab) == 129
    assert tokenizer.vocab[128] == "12"
    assert len(tokenizer.merge_rules) == 1
    assert tokenizer.merge_rules[(49, 50)] == 128


@max_score(3)
def test_two_merges():
    tokenizer = ASCIIBPETokenizer.from_data("123123", 2)
    ####### after two merges #######
    # "1" -> 49
    # "2" -> 50
    # "3" -> 51
    # "12" -> 128
    # "123" -> 129
    assert len(tokenizer.vocab) == 130
    assert tokenizer.vocab[129] == "123"
    assert len(tokenizer.merge_rules) == 2
    assert tokenizer.merge_rules[(49, 50)] == 128
    assert tokenizer.merge_rules[(128, 51)] == 129


@max_score(3)
def test_100_merges():
    """Train 100 merges on a real corpus and compare every rule.

    The corpus is vendored in tests/ rather than pulled from HuggingFace:
    karpathy/tiny_shakespeare is a script-based dataset and `datasets` dropped
    support for those, so the original version of this test errored out on any
    current install. Expected rules were regenerated from the vendored file.
    """
    here = Path(__file__).resolve().parent
    shakespear = (here / "tiny_shakespeare.txt").read_text()
    expected = {
        tuple(int(x) for x in k.split(",")): v
        for k, v in json.loads((here / "expected_merges_100.json").read_text()).items()
    }

    tokenizer = ASCIIBPETokenizer.from_data(shakespear, 100)
    assert tokenizer.merge_rules == expected
    assert len(tokenizer.vocab) == 228
