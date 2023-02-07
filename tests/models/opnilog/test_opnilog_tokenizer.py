# Third Party
import pytest

# Local
from opni_inference_service.models.opnilog.opnilog_tokenizer import LogTokenizer


@pytest.fixture
def tokenizer():
    return LogTokenizer(".test-tokenizer/")


def test_init(tokenizer):
    assert tokenizer.filepath == ".test-tokenizer/"
    assert tokenizer.n_words == 10000
    assert tokenizer.valid_words == 5


def test_addWord(tokenizer):
    tokenizer.addWord("test")
    assert tokenizer.word2index["test"] == 5
    assert tokenizer.index2word[5] == "test"
    assert tokenizer.valid_words == 6


def test_save_vocab(tokenizer):
    tokenizer.save_vocab()
    assert tokenizer.n_words == 10000
    assert tokenizer.valid_words == 5


def test_load_vocab(tokenizer):
    tokenizer.load_vocab()
    assert tokenizer.n_words == 10000
    assert tokenizer.valid_words == 5


def test_is_num_there(tokenizer):
    assert tokenizer.is_num_there("test12") == True
    assert tokenizer.is_num_there("test") == False


def test_tokenize(tokenizer):
    res = tokenizer.tokenize("<CLS> test", isTrain=True)
    assert res == [1, 5]
    assert tokenizer.word2index["test"] == 5
    assert tokenizer.index2word[5] == "test"
