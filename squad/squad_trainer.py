""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

""" Removing multi-gpu training and distributed training for simplicity. """


import os
import torch
import random
import timeit
import logging
import numpy as np

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from squad.squad_classes import SquadResult


from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.trainer_utils import is_main_process
from torch.utils.tensorboard import SummaryWriter

from squad.squad_loaders import load_and_cache_examples, load_graph


# Device Config
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class SquadTrainer:
    def __init__(self, args, model):
        self.args = args
        self.config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
            use_fast=False,
        )
        # self.model = AutoModelForQuestionAnswering.from_pretrained(
        # args.model_name_or_path, config=self.config,
        # )
        self.model = model

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        self.model.to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )

        self.train_dataset = None
        self.eval_dataset = None
        self.train_sampler = None
        self.eval_sampler = None
        self.train_dataloader = None
        self.eval_dataloader = None

        self.global_step = 0
        self.epoch = 0
        self.total_train_loss = 0.0
        self.tb_writer = None

        self.tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

    def train(self):
        args = self.args
        tb_writer = SummaryWriter()

        if args.do_train:
            train_dataset = load_and_cache_examples(
                args, self.tokenizer, evaluate=False, output_examples=False
            )
            train_graph = load_graph(args, evaluate=False)

        self.args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
        )

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps
                // (len(self.train_dataloader) // args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(self.train_dataloader)
                // self.args.gradient_accumulation_steps
                * self.args.num_train_epochs
            )

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total,
        )

        if os.path.isfile(
            os.path.join(args.model_name_or_path, "optimizer.pt")
        ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
            )

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info(
            "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
        )
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info(
            "  Gradient Accumulation steps = %d", args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 1
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if os.path.exists(args.model_name_or_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info(
                    "  Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", global_step)
                logger.info(
                    "  Will skip the first %d steps in the first epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("  Starting fine-tuning.")

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(
            epochs_trained,
            int(args.num_train_epochs),
            desc="Epoch",
            disable=args.local_rank not in [-1, 0],
        )
        # Added here for reproductibility
        set_seed(args)

        for _ in train_iterator:
            epoch_iterator = tqdm(
                train_dataloader,
                desc="Iteration",
                disable=args.local_rank not in [-1, 0],
            )
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                self.model.train()
                batch = tuple(t.to(args.device) for t in batch)

                # Retrieve Graph Structure from train_graph
                for bid in batch[8]:
                    graph_data = train_graph[int(bid.cpu().detach().item())]

                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                    "graph_data": graph_data,
                }

                if args.model_type in [
                    "xlm",
                    "roberta",
                    "distilbert",
                    "camembert",
                    "bart",
                    "longformer",
                ]:
                    del inputs["token_type_ids"]

                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                    if args.version_2_with_negative:
                        inputs.update({"is_impossible": batch[7]})
                    if hasattr(self.model, "config") and hasattr(
                        self.model.config, "lang2id"
                    ):
                        inputs.update(
                            {
                                "langs": (
                                    torch.ones(batch[0].shape, dtype=torch.int64)
                                    * args.lang_id
                                ).to(args.device)
                            }
                        )

                outputs = self.model(**inputs)
                # model outputs are always tuple in transformers (see doc)
                loss = outputs[0]

                if args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel (not distributed) training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), args.max_grad_norm
                    )

                    self.optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    # Log metrics
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        if args.local_rank == -1 and args.evaluate_during_training:
                            results = self.evaluate(args, self.model, self.tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar(
                                    "eval_{}".format(key), value, global_step
                                )
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / args.logging_steps,
                            global_step,
                        )
                        logging_loss = tr_loss

                    # Save model checkpoint
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(
                            args.output_dir, "checkpoint-{}".format(global_step)
                        )
                        # Take care of distributed/parallel training
                        model_to_save = (
                            self.model.module
                            if hasattr(self.model, "module")
                            else self.model
                        )
                        model_to_save.save_pretrained(output_dir)
                        self.tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(
                            self.optimizer.state_dict(),
                            os.path.join(output_dir, "optimizer.pt"),
                        )
                        torch.save(
                            scheduler.state_dict(),
                            os.path.join(output_dir, "scheduler.pt"),
                        )
                        logger.info(
                            "Saving optimizer and scheduler states to %s", output_dir
                        )

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

        tb_writer.close()

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        return global_step, tr_loss / global_step

    def evaluate(self, prefix=""):
        # TODO: Determine deprecation status
        args = self.args

        dataset, examples, features = load_and_cache_examples(
            args, self.tokenizer, evaluate=True, output_examples=True
        )
        eval_graph = load_graph(args, evaluate=True)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(
            dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_results = []
        start_time = timeit.default_timer()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if args.model_type in [
                    "xlm",
                    "roberta",
                    "distilbert",
                    "camembert",
                    "bart",
                    "longformer",
                ]:
                    del inputs["token_type_ids"]

                feature_indices = batch[3]

                # XLNet and XLM use more arguments for their predictions
                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                    # for lang_id-sensitive xlm models
                    if hasattr(self.model, "config") and hasattr(
                        self.model.config, "lang2id"
                    ):
                        inputs.update(
                            {
                                "langs": (
                                    torch.ones(batch[0].shape, dtype=torch.int64)
                                    * args.lang_id
                                ).to(args.device)
                            }
                        )
                outputs = self.model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [
                    (output[i].detach().cpu().tolist()) for output in outputs.to_tuple()
                ]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        evalTime = timeit.default_timer() - start_time
        logger.info(
            "  Evaluation done in total %f secs (%f sec per example)",
            evalTime,
            evalTime / len(dataset),
        )

        # Compute predictions
        output_prediction_file = os.path.join(
            args.output_dir, "predictions_{}.json".format(prefix)
        )
        output_nbest_file = os.path.join(
            args.output_dir, "nbest_predictions_{}.json".format(prefix)
        )

        if args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(
                args.output_dir, "null_odds_{}.json".format(prefix)
            )
        else:
            output_null_log_odds_file = None

        # XLNet and XLM use a more complex post-processing procedure
        if args.model_type in ["xlnet", "xlm"]:
            start_n_top = (
                self.model.config.start_n_top
                if hasattr(self.model, "config")
                else self.model.module.config.start_n_top
            )
            end_n_top = (
                self.model.config.end_n_top
                if hasattr(self.model, "config")
                else self.model.module.config.end_n_top
            )

            predictions = compute_predictions_log_probs(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                start_n_top,
                end_n_top,
                args.version_2_with_negative,
                self.tokenizer,
                args.verbose_logging,
            )
        else:
            predictions = compute_predictions_logits(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                args.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                args.verbose_logging,
                args.version_2_with_negative,
                args.null_score_diff_threshold,
                self.tokenizer,
            )

        # Compute the F1 and exact scores.
        results = squad_evaluate(examples, predictions)
        return results
