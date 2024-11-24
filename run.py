import glob
import logging
import os
import torch
import transformers
from model.graph_qa import GraphQA
from squad.squad_loaders import get_parser, load_and_cache_examples, load_graph
from squad.squad_trainer import evaluate, set_seed, train
from transformers import AutoConfig, AutoTokenizer, WEIGHTS_NAME

logger = logging.getLogger(__name__)

args = get_parser()

if args.doc_stride >= args.max_seq_length - args.max_query_length:
    logger.warning(
        "WARNING - You've set a doc stride which may be superior to the document length in some "
        "examples. This could result in errors when building features from the examples. Please reduce the doc "
        "stride or increase the maximum length to ensure the features are correctly built."
    )

if (
    os.path.exists(args.output_dir)
    and os.listdir(args.output_dir)
    and args.do_train
    and not args.overwrite_output_dir
):
    raise ValueError(
        "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
            args.output_dir
        )
    )

# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
args.device = device

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
)
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    args.local_rank,
    device,
    args.n_gpu,
    bool(args.local_rank != -1),
    args.fp16,
)

transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
# Set seed
set_seed(args)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

args.model_type = args.model_type.lower()
config = AutoConfig.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    cache_dir=args.cache_dir if args.cache_dir else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
    use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
)

model = GraphQA.from_pretrained(
    args.model_name_or_path,
    config=config,
    cache_dir=args.cache_dir if args.cache_dir else None,
)

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

model.to(args.device)

logger.info("Training/evaluation parameters %s", args)

# Training
if args.do_train:
    train_dataset = load_and_cache_examples(
        args, tokenizer, evaluate=False, output_examples=False
    )
    train_graph = load_graph(args, evaluate=False)
    # TODO(mingzhe): Merge these two parts
    global_step, tr_loss = train(args, train_dataset, train_graph, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

# Save the trained model and the tokenizer
if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = GraphQA.from_pretrained(args.output_dir)

    # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    # So we use use_fast=False here for now until Fast-tokenizer-compatible-examples are out
    tokenizer = AutoTokenizer.from_pretrained(
        args.output_dir, do_lower_case=args.do_lower_case, use_fast=False
    )
    model.to(args.device)

# Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
results = {}
if args.do_eval and args.local_rank in [-1, 0]:
    if args.do_train:
        logger.info("Loading checkpoints saved during training for evaluation")
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
                )
            )
    else:
        logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
        checkpoints = [args.model_name_or_path]

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        # Reload the model
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        model = GraphQA.from_pretrained(checkpoint)
        model.to(args.device)

        # Evaluate
        result = evaluate(args, model, tokenizer, prefix=global_step)

        logger.info("results: {}".format(result))

        result = dict(
            (k + ("_{}".format(global_step) if global_step else ""), v)
            for k, v in result.items()
        )
        results.update(result)

logger.info("Results: {}".format(results))
