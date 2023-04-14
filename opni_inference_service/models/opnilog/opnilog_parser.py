# Standard Library
import logging
import os
import time
from copy import deepcopy

# Third Party
import pandas as pd
import torch
import torch.nn as nn
from const import THRESHOLD
from torchvision import transforms
from utils import put_model_stats

# Local
from models.opnilog.opnilog_model import *  # should improve this
from models.opnilog.opnilog_tokenizer import LogTokenizer

# constant
# tell torch using or not using GPU
using_GPU = True if torch.cuda.device_count() > 0 else False
if not using_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")


class LogParser:
    def __init__(
        self,
        k=50,
        log_format="<Content>",
        model_name="nulog_model_latest.pt",
        save_path="output/",
    ):
        self.savePath = save_path
        self.k = k
        self.df_log = None
        self.log_format = log_format
        self.tokenizer = LogTokenizer(self.savePath)

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        self.model_name = model_name
        self.model_path = os.path.join(self.savePath, self.model_name)

    def num_there(self, s):
        digits = [i.isdigit() for i in s]
        return True if np.mean(digits) > 0.0 else False

    def save_model(self, model, model_opt, epoch, loss):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model_opt.state_dict(),
                "loss": loss,
            },
            self.model_path,
        )

    def load_model(self, model, model_opt):
        if using_GPU:
            ckpt = torch.load(self.model_path)
        else:
            ckpt = torch.load(self.model_path, map_location=torch.device("cpu"))
        try:
            model_opt.load_state_dict(ckpt["optimizer_state_dict"])
            model.load_state_dict(ckpt["model_state_dict"])
            epoch = ckpt["epoch"]
            loss = ckpt["loss"]
            return epoch, loss
        except:  ## TODO: remove this try except when we use the new save function.
            logging.warning("loading trained model with old format.")
            model.load_state_dict(ckpt)

    def train(
        self,
        data_tokenized,
        batch_size=32,
        mask_percentage=1.0,
        pad_len=64,
        N=1,
        d_model=256,
        dropout=0.1,
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.005,
        nr_epochs=5,
        num_samples=0,
        step_size=100,
        put_results=False,
        is_streaming=False,
        iter_function=None,
        iter_input_list=None,
    ):
        self.mask_percentage = mask_percentage
        self.pad_len = pad_len
        self.batch_size = batch_size
        self.N = N
        self.d_model = d_model
        self.dropout = dropout
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.num_samples = num_samples
        self.nr_epochs = nr_epochs
        self.step_size = step_size
        self.is_streaming = is_streaming

        logging.debug("learning rate : " + str(self.lr))
        transform_to_tensor = transforms.Lambda(lambda lst: torch.tensor(lst))

        criterion = nn.CrossEntropyLoss()
        model = self.make_model(
            self.tokenizer.n_words,
            self.tokenizer.n_words,
            N=self.N,
            d_model=self.d_model,
            d_ff=self.d_model,
            dropout=self.dropout,
            max_len=self.pad_len,
        )

        if using_GPU:
            model.cuda()
        model_opt = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

        if (self.model_name) in os.listdir(self.savePath):
            # model.load_state_dict(torch.load(self.model_path))
            prev_epoch, prev_loss = self.load_model(model, model_opt)

        if self.is_streaming:
            train_dataloader = self.get_streaming_dataloader(
                iter_function, iter_input_list
            )
            self.training_batch_size = (
                self.num_samples // self.batch_size
            )  # streaming dataloader has no len()
        else:
            train_dataloader, eval_dataloader = self.get_train_eval_dataloaders(
                data_tokenized, transform_to_tensor
            )
            self.training_batch_size = len(train_dataloader)
        ## train if no model
        model.train()
        logging.info(f"#######Training Model within {self.nr_epochs} epochs...######")
        training_start_time = time.time()
        for epoch in range(self.nr_epochs):
            self.run_epoch(
                train_dataloader,
                model,
                SimpleLossCompute(model.generator, criterion, model_opt),
                epoch=epoch,
                training_start_time=training_start_time,
                put_results=put_results,
                is_streaming=is_streaming,
            )
        end_time = time.time()
        if put_results:
            put_model_stats(model_status="completed", statistics={})

        self.save_model(model=model, model_opt=model_opt, epoch=self.nr_epochs, loss=0)

        if not self.is_streaming:
            # eval in validation dataset
            self.init_inference()
            eval_predictions = self.predict(eval_dataloader, False)
            num_predictions = len(eval_predictions)
            num_normal = 0
            for pred in eval_predictions:
                if pred >= THRESHOLD:
                    num_normal += 1
            logging.info(
                f"Model finished training predicting {num_normal} logs correctly out of {num_predictions} total logs for an accuracy of {num_normal/num_predictions} on eval dataset."
            )

    def init_inference(
        self,
        batch_size=32,
        mask_percentage=1.0,
        pad_len=64,
        N=1,
        d_model=256,
        dropout=0.1,
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=0.005,
        nr_epochs=5,
        num_samples=0,
        step_size=100,
    ):
        # training can share this init function
        self.mask_percentage = mask_percentage
        self.pad_len = pad_len
        self.batch_size = batch_size
        self.N = N
        self.d_model = d_model
        self.dropout = dropout
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.num_samples = num_samples
        self.nr_epochs = nr_epochs
        self.step_size = step_size

        # data_tokenized = data_tokenized[:10]
        self.transform_to_tensor = transforms.Lambda(lambda lst: torch.tensor(lst))

        self.criterion = nn.CrossEntropyLoss()
        self.model = self.make_model(
            self.tokenizer.n_words,
            self.tokenizer.n_words,
            N=self.N,
            d_model=self.d_model,
            d_ff=self.d_model,
            dropout=self.dropout,
            max_len=self.pad_len,
        )

        if using_GPU:
            self.model.cuda()
        self.model_opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        self.load_model(self.model, self.model_opt)
        self.model.eval()

    def predict(self, data_tokenized, is_array=True, output_prefix=""):
        if is_array:
            test_dataloader = self.get_test_dataloaders(
                data_tokenized, self.transform_to_tensor
            )
        else:
            test_dataloader = data_tokenized

        results = self.run_test(
            test_dataloader,
            self.model,
            SimpleLossCompute(self.model.generator, self.criterion, None, is_test=True),
        )

        anomaly_preds = []
        for i, (x, y, ind) in enumerate(results):
            true_pred = 0
            total_count = 0
            c_rates = []
            for j in range(len(x)):
                if not self.num_there(self.tokenizer.index2word[y[j]]):

                    if y[j] in x[j][-self.k :]:  ## if it's within top k predictions
                        true_pred += 1
                    total_count += 1

                if j == len(x) - 1 or ind[j] != ind[j + 1]:
                    this_rate = 1.0 if total_count == 0 else true_pred / total_count
                    c_rates.append(this_rate)
                    true_pred = 0
                    total_count = 0

            anomaly_preds.extend(c_rates)

        return anomaly_preds

    def get_streaming_dataloader(self, iter_function, iter_input_list):
        """
        the streaming dataloader simply incorporate the IterablePaddedDataset,
        the number of workers will depends on the amount of items in the iter_input_list
        """
        num_workers = len(iter_input_list)
        train_data = IterablePaddedDataset(
            tokenizer=self.tokenizer,
            iter_function=iter_function,
            iter_input_list=iter_input_list,
            pad_len=self.pad_len,
        )
        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            num_workers=num_workers,
        )
        return train_dataloader

    def get_train_eval_dataloaders(
        self, data_tokenized, transform_to_tensor, training_eval_split=0.9
    ):
        train_data = MaskedDataset(
            data_tokenized,
            self.tokenizer,
            mask_percentage=self.mask_percentage,
            transforms=transform_to_tensor,
            pad_len=self.pad_len,
        )
        weights = train_data.get_sample_weights()
        if self.num_samples != 0:
            all_data_sampler = WeightedRandomSampler(
                weights=list(weights), num_samples=self.num_samples, replacement=True
            )
        else:
            all_data_sampler = RandomSampler(train_data)
        all_data_sampler_list = list(all_data_sampler)
        train_eval_index = int(training_eval_split * len(all_data_sampler_list))
        train_sampler = all_data_sampler_list[:train_eval_index]
        eval_sampler = all_data_sampler_list[train_eval_index:]

        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.batch_size
        )
        eval_dataloader = DataLoader(
            train_data, sampler=eval_sampler, batch_size=self.batch_size
        )
        return train_dataloader, eval_dataloader

    def get_test_dataloaders(self, data_tokenized, transform_to_tensor):
        test_data = MaskedDataset(
            data_tokenized,
            self.tokenizer,
            mask_percentage=self.mask_percentage,
            transforms=transform_to_tensor,
            pad_len=self.pad_len,
        )
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data, sampler=test_sampler, batch_size=self.batch_size
        )
        return test_dataloader

    def load_data(self, windows_folder_path):
        try:
            df_log = self.log_to_dataframe(windows_folder_path)
            return df_log[~df_log["Content"].isnull()]["Content"].tolist()
        except Exception as e:
            logging.error("Unable to fetch data.")
            return []

    def tokenize_data(self, input_text, is_training=False):
        return self.tokenizer.tokenize_data(input_text, is_training=is_training)

    def log_to_dataframe(self, windows_folder_path):
        """Function to transform log file to dataframe"""
        all_log_messages = []
        json_files = sorted(
            file
            for file in os.listdir(windows_folder_path)
            if file.endswith(".json.gz")
        )
        for window_file in json_files:
            window_df = pd.read_json(
                os.path.join(windows_folder_path, window_file), lines=True
            )
            masked_log_messages = window_df["masked_log"]
            for index, message in masked_log_messages.items():
                all_log_messages.append([message])

        logdf = pd.DataFrame(all_log_messages, columns=["Content"])
        logdf.insert(0, "LineId", None)
        logdf["LineId"] = [i + 1 for i in range(len(all_log_messages))]
        return logdf

    def do_mask(self, batch):
        c = copy.deepcopy
        token_id = self.tokenizer.word2index["<MASK>"]
        srcs, offsets, data_lens, indices = batch
        src, trg, idxs = [], [], []

        for i, _ in enumerate(data_lens):
            data_len = c(data_lens[i].item())

            dg = c(indices[i].item())
            masked_data = c(srcs[i])
            offset = offsets[i].item()
            num_masks = round(self.mask_percentage * data_len)
            if self.mask_percentage < 1.0:
                masked_indices = np.random.choice(
                    np.arange(offset, offset + data_len),
                    size=num_masks if num_masks > 0 else 1,
                    replace=False,
                )
            else:
                masked_indices = np.arange(offset, offset + data_len)

            masked_indices.sort()

            for j in masked_indices:
                tmp = c(masked_data)
                label_y = c(tmp[j])
                tmp[j] = token_id
                src.append(c(tmp))

                trg.append(label_y)
                idxs.append(dg)
        return torch.stack(src), torch.stack(trg), torch.Tensor(idxs)

    def get_padded_data(self, raw_data, is_training=True):
        """
        tokenize the streaming data and add padding tokens to the pad_len. Only invoked in streaming.
        """
        data = self.tokenizer.tokenize_data(raw_data, is_training=is_training)
        pad_len = self.pad_len
        d = deepcopy(data)
        npd = np.asarray(d)
        pd = np.zeros(shape=(len(d), pad_len))
        for n in range(len(d)):
            if len(npd[n]) > pad_len:
                pd[n] = np.asarray(npd[n][:pad_len])
            else:
                pd[n][: len(npd[n])] = np.asarray(npd[n])
        pd = pd.astype("long")
        return self.prepare_metadata(pd, d)

    def prepare_metadata(self, padded_data, data):
        """
        prepare metadata for do_mask(), includes the source tokens, origin data_len, its indices and offsets.
        """
        srcs = []
        offsets = []
        data_lens = []
        indices = []
        for index, src in enumerate(padded_data):

            offset = 1
            data_len = (
                len(data[index]) - 1
                if len(data[index]) < self.pad_len
                else self.pad_len - 1
            )

            srcs.append(src)
            offsets.append(offset)
            data_lens.append(data_len)
            indices.append(index)
        return (
            torch.tensor(np.array(srcs)),
            torch.tensor(np.array(offsets)),
            torch.tensor(np.array(data_lens)),
            torch.tensor(np.array(indices)),
        )

    def run_epoch(
        self,
        dataloader,
        model,
        loss_compute,
        epoch,
        training_start_time,
        put_results,
        is_streaming=False,
    ):

        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(dataloader):
            if is_streaming:
                batch = self.get_padded_data(batch, is_training=True)
            b_input, b_labels, _ = self.do_mask(batch)
            batch = Batch(b_input, b_labels, 0)
            if using_GPU:
                out = model.forward(
                    batch.src.cuda(),
                    batch.trg.cuda(),
                    batch.src_mask.cuda(),
                    batch.trg_mask.cuda(),
                )

                loss = loss_compute(out, batch.trg_y.cuda(), batch.ntokens)
            else:
                out = model.forward(
                    batch.src, batch.trg, batch.src_mask, batch.trg_mask
                )

                loss = loss_compute(out, batch.trg_y, batch.ntokens)

            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens

            if i % self.step_size == 1:
                elapsed = time.time() - start
                training_progress = (
                    (i / self.training_batch_size) + epoch
                ) / self.nr_epochs
                total_time_taken = time.time() - training_start_time
                remaining_time = (
                    total_time_taken // training_progress
                ) - total_time_taken
                logging.info(
                    f"| Epoch: {epoch} | Total Progress: {(training_progress * 100):.2f}% | Training Time Taken: {total_time_taken:.2f}s | ETC: {(remaining_time):.2f}s | Epoch Step: {i}/{self.training_batch_size} | Loss: {(loss / batch.ntokens):.4f} | Tokens per Sec: {(tokens / elapsed):.2f} |"
                )
                if put_results:
                    statistics_dict = {
                        "stage": "train",
                        "percentageCompleted": int(100 * training_progress),
                        "timeElapsed": int(total_time_taken),
                        "remainingTime": int(remaining_time),
                        "currentEpoch": epoch + 1,
                    }
                    put_model_stats(
                        model_status="training",
                        statistics=statistics_dict,
                    )
                start = time.time()
                tokens = 0
        return total_loss / total_tokens

    def run_test(self, dataloader, model, loss_compute):
        # Standard Library
        import time

        model.eval()
        if using_GPU:
            model.cuda()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                b_input, b_labels, ind = self.do_mask(batch)

                batch = Batch(b_input, b_labels, 0)
                if using_GPU:
                    out = model.forward(
                        batch.src.cuda(),
                        batch.trg.cuda(),
                        batch.src_mask.cuda(),
                        batch.trg_mask.cuda(),
                    )
                else:
                    out = model.forward(
                        batch.src, batch.trg, batch.src_mask, batch.trg_mask
                    )

                out_p = model.generator(out)  # batch_size, hidden_dim
                t3 = time.perf_counter()

                if i % self.step_size == 1:
                    logging.debug("Epoch Step: %d / %d" % (i, len(dataloader)))
                # r1 = out_p.cpu().numpy().argsort(axis=1) # this is why it's so slow
                r11 = torch.argsort(out_p, dim=1)
                r1 = r11.cpu().numpy()
                r2 = b_labels.cpu().numpy()
                r3 = ind.cpu()
                yield r1, r2, r3
                t4 = time.perf_counter()

    def make_model(
        self,
        src_vocab,
        tgt_vocab,
        N=3,
        d_model=512,
        d_ff=2048,
        h=8,
        dropout=0.1,
        max_len=20,
    ):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout, max_len)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab),
        )

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        return model
