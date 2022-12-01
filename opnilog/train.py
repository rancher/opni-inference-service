# Standard Library
import json
import logging

# Third Party
from models.opnilog.masker import LogMasker
from models.opnilog.opnilog_parser import LogParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")


def load_text(training_file, duplicate=1):
    texts = []
    masker = LogMasker()
    with open(training_file) as fin:
        for idx, line in enumerate(fin):
            try:
                log = json.loads(line)
                log = log["log"]
                log = masker.mask(log)
                texts.append(log)
            except Exception as e:
                logging.error(e)

    texts = texts * duplicate
    print(texts)
    return texts


def train_opnilog_model(training_file, duplicate=1):
    nr_epochs = 3
    num_samples = 0
    parser = LogParser()
    texts = load_text(training_file, duplicate)

    tokenized = parser.tokenize_data(texts, isTrain=True)
    parser.tokenizer.save_vocab()
    parser.train(tokenized, nr_epochs=nr_epochs, num_samples=num_samples)


if __name__ == "__main__":
    train_opnilog_model(training_file="input/mix-raw.log")
