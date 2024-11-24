import os
import json
import jsonpickle
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from transformers.utils import is_torch_available, logging
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    TruncationStrategy,
)


from components.hg_parser import ConstituencyParser, ConstituencyNode
from squad.squad_classes import SquadFeatures

import torch
from torch.utils.data import TensorDataset

MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}


logger = logging.get_logger(__name__)


def _improve_answer_span(
    doc_tokens, input_start, input_end, tokenizer, orig_answer_text
):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def squad_convert_example_to_features(
    tokenizer,
    example,
    max_seq_length,
    doc_stride,
    max_query_length,
    padding_strategy,
    is_training,
):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning(
                f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'"
            )
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for i, token in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens,
            tok_start_position,
            tok_end_position,
            tokenizer,
            example.answer_text,
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_query_length,
    )

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_sentences_pair
    )

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length
            - doc_stride
            - len(truncated_query)
            - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][
                    : encoded_dict["input_ids"].index(tokenizer.pad_token_id)
                ]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"])
                    - 1
                    - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][
                    last_padding_id_position + 1 :
                ]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = (
                len(truncated_query) + sequence_added_tokens + i
                if tokenizer.padding_side == "right"
                else i
            )
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = (
            len(truncated_query) + sequence_added_tokens
        )
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict
            and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _check_is_max_context(
                spans, doc_span_index, doc_span_index * doc_stride + j
            )
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"]
                + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        sep_index = span["input_ids"].index(tokenizer.sep_token_id)

        # p_mask: mask with 1 for token that cannot be in the answer (0 for token which can be in an answer)
        # Original TF implementation also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            p_mask[
                -len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)
            ] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(
                span["input_ids"], already_has_special_tokens=True
            )
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                sep_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                doc_tokens=span_doc_tokens,
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            )
        )
    return features


def squad_convert_example_to_features_init(
    tokenizer_for_convert: PreTrainedTokenizerBase,
):
    global tokenizer
    tokenizer = tokenizer_for_convert


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model. It is
    model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of [`~data.processors.squad.SquadExample`]
        tokenizer: an instance of a child of [`PreTrainedTokenizer`]
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        padding_strategy: Default to "max_length". Which padding strategy to use
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset, if 'tf': returns a tf.data.Dataset
        threads: multiple processing threads.


    Returns:
        list of [`~data.processors.squad.SquadFeatures`]

    Example:

    ```python
    processor = SquadV2Processor()
    examples = processor.get_dev_examples(data_dir)

    features = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,
    )
    ```"""
    # Defining helper methods
    features = []

    threads = min(threads, cpu_count())
    with Pool(
        threads,
        initializer=squad_convert_example_to_features_init,
        initargs=(tokenizer,),
    ) as p:
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,
            )
        )

    # Add example index and unique id
    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in tqdm(
        features,
        total=len(features),
        desc="add example index and unique id",
        disable=not tqdm_enabled,
    ):
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    # Constituency Graph Construction
    # Stanford NLP Parser doesn‘t support multiprocessing mode, so that we need to operate it with the single track.
    c_parser = ConstituencyParser()
    graph_f = open("./data/squad_v2/graph/v1", "w")
    graph_ids = list()
    graph_id = 0

    for feature in tqdm(features, desc="constituency graph construction"):
        # 0) Meta info
        start_position = feature.start_position
        end_position = feature.end_position
        # qas_id = feature.qas_id
        context = tokenizer.decode(
            feature.input_ids[feature.sep_index + 1 :], skip_special_tokens=True
        )
        constituents = c_parser.get_sentences(context)

        # 1) token2token feature mapping dict
        token2token_mapping = dict()
        for index, token in enumerate(feature.doc_tokens):
            token2token_mapping[index] = feature.sep_index + index

        # 2) leaf2token feature mapping dict
        try:
            leaf_id = 0
            last_token_id = feature.sep_index + 1
            leaf_node_mapping = dict()

            for constituent in constituents:
                leaves = constituent.constituency.leaf_labels()
                for leaf in leaves:
                    leaf = leaf.lower()
                    current_token_text = ""
                    current_token_ids = list()
                    while current_token_text != leaf:
                        current_token_text += (
                            feature.tokens[last_token_id]
                            if not feature.tokens[last_token_id].startswith("##")
                            else feature.tokens[last_token_id][2:]
                        )
                        current_token_ids += [last_token_id]
                        last_token_id += 1
                    leaf_node_mapping[leaf_id] = current_token_ids
                    leaf_id += 1
        except Exception as e:
            # If the preterminals are different with tokens
            print("Constituents Leaf Parser Failed, Skip this one: {e}")
            graph_ids += [-1]
            continue

        # 3) constituent2leaf feature mapping dict
        try:
            pid, cid = 0, 1000000000
            R_cc, R_ct = list(), list()

            def iterate_tree(root):
                nonlocal pid, cid
                if root.is_preterminal():
                    tids = leaf_node_mapping[pid]
                    is_answer = False
                    if tids[0] == start_position and tids[-1] == end_position:
                        is_answer = True
                    leaf_node = ConstituencyNode(
                        cid=pid,
                        label=root.label,
                        text=root.leaf_labels(),
                        lids=[pid],
                        tids=tids,
                        children=[],
                        is_answer=is_answer,
                    )
                    R_ct += [[pid, tid] for tid in tids]
                    pid += 1
                    return leaf_node
                else:
                    child_nodes, lids, tids = list(), list(), list()
                    for child in root.children:
                        child_node = iterate_tree(child)
                        child_nodes += [child_node]
                        lids += child_node.lids
                        for lid in child_node.lids:
                            tids += leaf_node_mapping[lid]
                    is_answer = False
                    if tids[0] == start_position and tids[-1] == end_position:
                        is_answer = True
                    leaf_node = ConstituencyNode(
                        cid=cid,
                        label=root.label,
                        text=root.leaf_labels(),
                        lids=lids,
                        tids=tids,
                        children=child_nodes,
                        is_answer=is_answer,
                    )
                    R_cc += [[cid, cid_] for cid_ in child_nodes]
                    R_ct += [[cid, tid] for tid in tids]
                    cid += 1
                    return leaf_node

            child_nodes, lids, tids = list(), list(), list()

            for constituent in constituents:
                root = iterate_tree(constituent.constituency)
                child_nodes += [root]
                lids += root.lids
                for lid in root.lids:
                    tids += leaf_node_mapping[lid]

            c_graph_node = ConstituencyNode(
                cid=cid,
                label="CONTEXT",
                text=context,
                lids=lids,
                tids=tids,
                children=child_nodes,
                is_answer=False,
            )

            c_edge_index = {
                ("token", "connect", "constituent"): torch.tensor(R_ct)
                .t()
                .contiguous(),
                ("constituent", "connect", "constituent"): torch.tensor(R_cc)
                .t()
                .contiguous(),
            }

            # For debugging
            # ConstituencyNode.iterate(c_graph_node)
            graph_f.write(
                jsonpickle.encode(
                    {
                        "qid": graph_id,
                        "graph": c_graph_node,
                        "edge_index": c_edge_index,
                    },
                    indent=None,
                )
                + "\n"
            )
            graph_ids += [graph_id]
            graph_id += 1

        except Exception as e:
            print(f"Constituents Token Parser Failed, Skip This One: {e}")
            graph_ids += [-1]
            continue

    graph_f.close()

    if return_dataset == "pt":
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        all_token_type_ids = torch.tensor(
            [f.token_type_ids for f in features], dtype=torch.long
        )
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor(
            [f.is_impossible for f in features], dtype=torch.float
        )
        all_qas_id = torch.tensor(graph_ids, dtype=torch.long)

        if not is_training:
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_feature_index,
                all_cls_index,
                all_p_mask,
                all_qas_id,
            )
        else:
            all_start_positions = torch.tensor(
                [f.start_position for f in features], dtype=torch.long
            )
            all_end_positions = torch.tensor(
                [f.end_position for f in features], dtype=torch.long
            )
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
                all_qas_id,
            )

        return features, dataset
    else:
        return features
