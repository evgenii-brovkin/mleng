import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from transformers import *
import tokenizers

import argparse
import os


def opj(*args):
    return os.path.normpath(os.path.join(*args))


def build_model(MAX_LEN, PATH):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(opj(PATH, "config-roberta-base.json"))
    bert_model = TFRobertaModel.from_pretrained(
        opj(PATH, "pretrained-roberta-base.h5"), config=config
    )
    x = bert_model(ids, attention_mask=att, token_type_ids=tok)

    x1 = tf.keras.layers.Dropout(0.1)(x[0])
    x1 = tf.keras.layers.Conv1D(1, 1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation("softmax")(x1)

    x2 = tf.keras.layers.Dropout(0.1)(x[0])
    x2 = tf.keras.layers.Conv1D(1, 1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation("softmax")(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1, x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    return model


def main(args):
    MAX_LEN = 96
    models_path = args.models_path
    data_path = args.data_path
    output_path = args.output_path

    tokenizer = tokenizers.ByteLevelBPETokenizer(
        vocab_file=opj(models_path, "vocab-roberta-base.json"),
        merges_file=opj(models_path, "merges-roberta-base.txt"),
        lowercase=True,
        add_prefix_space=True,
    )
    sentiment_id = {"positive": 1313, "negative": 2430, "neutral": 7974}
    train = pd.read_csv(opj(data_path, "train.csv")).fillna("").head()

    ct = train.shape[0]
    input_ids = np.ones((ct, MAX_LEN), dtype="int32")
    attention_mask = np.zeros((ct, MAX_LEN), dtype="int32")
    token_type_ids = np.zeros((ct, MAX_LEN), dtype="int32")
    start_tokens = np.zeros((ct, MAX_LEN), dtype="int32")
    end_tokens = np.zeros((ct, MAX_LEN), dtype="int32")

    for k in range(train.shape[0]):
        text1 = " " + " ".join(train.loc[k, "text"].split())
        text2 = " ".join(train.loc[k, "selected_text"].split())
        idx = text1.find(text2)
        chars = np.zeros((len(text1)))
        chars[idx : idx + len(text2)] = 1
        if text1[idx - 1] == " ":
            chars[idx - 1] = 1
        enc = tokenizer.encode(text1)

        offsets = []
        idx = 0
        for t in enc.ids:
            w = tokenizer.decode([t])
            offsets.append((idx, idx + len(w)))
            idx += len(w)

        toks = []
        for i, (a, b) in enumerate(offsets):
            sm = np.sum(chars[a:b])
            if sm > 0:
                toks.append(i)

        s_tok = sentiment_id[train.loc[k, "sentiment"]]
        input_ids[k, : len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        attention_mask[k, : len(enc.ids) + 5] = 1
        if len(toks) > 0:
            start_tokens[k, toks[0] + 1] = 1
            end_tokens[k, toks[-1] + 1] = 1

    test = pd.read_csv(opj(data_path, "test.csv")).fillna("").head()

    ct = test.shape[0]
    input_ids_t = np.ones((ct, MAX_LEN), dtype="int32")
    attention_mask_t = np.zeros((ct, MAX_LEN), dtype="int32")
    token_type_ids_t = np.zeros((ct, MAX_LEN), dtype="int32")

    for k in range(ct):
        text1 = " " + " ".join(test.loc[k, "text"].split())
        enc = tokenizer.encode(text1)
        s_tok = sentiment_id[test.loc[k, "sentiment"]]
        input_ids_t[k, : len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        attention_mask_t[k, : len(enc.ids) + 5] = 1

    DISPLAY = 1  # USE display=1 FOR INTERACTIVE
    preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0], MAX_LEN))
    model = build_model(MAX_LEN, models_path)
    model.load_weights(opj(models_path, "v0-roberta-0.h5"))
    preds = model.predict(
        [input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY
    )
    preds_start += preds[0]
    preds_end += preds[1]

    all = []
    for k in range(input_ids_t.shape[0]):
        a = np.argmax(preds_start[k,])
        b = np.argmax(preds_end[k,])
        if a > b:
            st = test.loc[k, "text"]
        else:
            text1 = " " + " ".join(test.loc[k, "text"].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a - 1 : b])
        all.append(st)

    test["selected_text"] = all
    test[["textID", "selected_text"]].to_csv(opj(output_path, "submission.csv"), index=False)


def runQnA():
    from transformers.pipelines import pipeline

    model_name = "deepset/roberta-base-squad2"
    model_name = "distilbert-base-cased-distilled-squad"
    # model_name = "https://huggingface.co/deepset/roberta-base-squad2"
    # Get predictions
    print('Building pipeline...', end='')
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    print('Done')

    old_c = None
    print('Ready for questions!')
    while True:
        q = input('Input your question:')
        if q == 'exit':
            exit(0)
        
        c = input('Input context:')
        if c == 'exit':
            exit(0)
        
        if q == 'test' and c == 'test':
            res = nlp({
                'question': 'Why is model conversion important?',
                'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
            })
            print(f'Answer: {res["answer"]}')
            continue
        
        if c == '':
            if old_c is not None:
                c = old_c
            else:
                print('No previous context available')
                continue
        res = nlp({
            'question': q,
            'context': c
        })
        old_c = c
        print(f'Answer: {res["answer"]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", "-m", default="./models")
    parser.add_argument("--data_path", "-d", default="./data")
    parser.add_argument("--output_path", "-o", default="./output")
    args = parser.parse_args()
    # main(args)
    while True:
        try:
            runQnA()
        except KeyboardInterrupt:
            print('My job here is done.')
            break
        except KeyError:
            print('I do not have the answer for that.')


