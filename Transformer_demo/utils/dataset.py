import tensorflow_datasets as tfds
import tensorflow as tf
import os
MAX_LENGTH = 40
BUFFER_SIZE = 20000
BATCH_SIZE = 64

    
def init_tokenizer(train_examples):
    global tokenizer_en,tokenizer_pt
    print(os.getcwd())
    if os.path.exists("ted_hrlr_eng_vocab.subwords") and os.path.exists("ted_hrlr_pt_vocab.subwords"):
            tokenizer_en=tfds.deprecated.text.SubwordTextEncoder.load_from_file("ted_hrlr_eng_vocab")
            tokenizer_pt=tfds.deprecated.text.SubwordTextEncoder.load_from_file("ted_hrlr_pt_vocab")
    else:
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)
    return tokenizer_en,tokenizer_pt
def encode(lang1, lang2):
   
    global tokenizer_en,tokenizer_pt
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size+1]
    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size+1]
    return lang1, lang2
def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])

    return result_pt, result_en

def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)

def make_tran_valid_sets(train_examples,val_examples):
    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)
    return train_dataset,val_dataset