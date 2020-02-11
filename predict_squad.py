from __future__ import absolute_import, division, print_function

import copy
import functools
import json
import math
import os

import tensorflow as tf
from absl import app, flags, logging

import input_pipeline
import squad_lib
import tokenization
import tf_utils
from albert import AlbertConfig, AlbertModel

flags.DEFINE_enum(
    'mode', 'predict',
    ['train_and_predict', 'train', 'predict'],
    'One of {"train_and_predict", "train", "predict"}. '
    '`train_and_predict`: both train and predict to a json file. '
    '`train`: only trains the model. '
    '`predict`: predict answers from the squad json file. ')

# Predict processing related.
flags.DEFINE_string('predict_file', None,
                    'Prediction data path with train tfrecords.')
flags.DEFINE_integer('predict_batch_size', 8,
                     'Total batch size for predicting.')

flags.DEFINE_integer(
    'n_best_size', 20,
    'The total number of n-best predictions to generate in the '
    'nbest_predictions.json output file.')

flags.DEFINE_integer(
    'max_answer_length', 30,
    'The maximum length of an answer that can be generated. This is needed '
    'because the start and end predictions are not conditioned on one another.')

flags.DEFINE_integer(
    'doc_stride', 128,
    'doc_stride'
    'doc_stride')

flags.DEFINE_integer(
    'max_query_length', 64,
    'max_query_length'
    'max_query_length')
    
flags.DEFINE_string(
    "albert_config_file", 'config.json',
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")


flags.DEFINE_string("spm_model_file", '30k-clean.model',
                    "The model file for sentence piece tokenization.")

flags.DEFINE_string(
    "model_dir", '',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "checkpoint_path", 'ctl_step_6992.ckpt-3',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_enum(
    "strategy_type", "one", ["one", "mirror"],
    "Training strategy for single or multi gpu training")

# Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained ALBERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("start_n_top", default=5,
                     help="Beam size for span start.")

flags.DEFINE_integer("end_n_top", default=5, help="Beam size for span end.")


flags.DEFINE_bool(
    "version_2_with_negative", True,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")

flags.DEFINE_integer("seed", 42, "random_seed")


FLAGS = flags.FLAGS

class ALBertQALayer(tf.keras.layers.Layer):
    """Layer computing position and is_possible for question answering task."""

    def __init__(self, hidden_size, start_n_top, end_n_top, initializer, dropout, **kwargs):
        """Constructs Summarization layer.
        Args:
          hidden_size: Int, the hidden size.
          start_n_top: Beam size for span start.
          end_n_top: Beam size for span end.
          initializer: Initializer used for parameters.
          dropout: float, dropout rate.
          **kwargs: Other parameters.
        """
        super(ALBertQALayer, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.start_n_top = start_n_top
        self.end_n_top = end_n_top
        self.initializer = initializer
        self.dropout = dropout

    def build(self, unused_input_shapes):
        """Implements build() for the layer."""
        self.start_logits_proj_layer = tf.keras.layers.Dense(
            units=1, kernel_initializer=self.initializer, name='start_logits/dense')
        self.end_logits_proj_layer0 = tf.keras.layers.Dense(
            units=self.hidden_size,
            kernel_initializer=self.initializer,
            activation=tf.nn.tanh,
            name='end_logits/dense_0')
        self.end_logits_proj_layer1 = tf.keras.layers.Dense(
            units=1, kernel_initializer=self.initializer, name='end_logits/dense_1')
        self.end_logits_layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, name='end_logits/LayerNorm')
        self.answer_class_proj_layer0 = tf.keras.layers.Dense(
            units=self.hidden_size,
            kernel_initializer=self.initializer,
            activation=tf.nn.tanh,
            name='answer_class/dense_0')
        self.answer_class_proj_layer1 = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=self.initializer,
            use_bias=False,
            name='answer_class/dense_1')
        self.ans_feature_dropout = tf.keras.layers.Dropout(rate=self.dropout)
        super(ALBertQALayer, self).build(unused_input_shapes)

    def __call__(self,
                 sequence_output,
                 p_mask,
                 cls_index,
                 start_positions=None,
                 **kwargs):
        inputs = tf_utils.pack_inputs(
            [sequence_output, p_mask, cls_index, start_positions])
        return super(ALBertQALayer, self).__call__(inputs, **kwargs)


    def call(self, inputs, **kwargs):
        """Implements call() for the layer."""
        unpacked_inputs = tf_utils.unpack_inputs(inputs)
        sequence_output = unpacked_inputs[0]
        p_mask = unpacked_inputs[1]
        cls_index = unpacked_inputs[2]
        start_positions = unpacked_inputs[3]

        _, seq_len, _ = sequence_output.shape.as_list()
        sequence_output = tf.transpose(sequence_output, [1, 0, 2])

        start_logits = self.start_logits_proj_layer(sequence_output)
        start_logits = tf.transpose(tf.squeeze(start_logits, -1), [1, 0])
        start_logits_masked = start_logits * (1 - p_mask) - 1e30 * p_mask
        start_log_probs = tf.nn.log_softmax(start_logits_masked, -1)

        if kwargs.get("training", False):
            # during training, compute the end logits based on the
            # ground truth of the start position
            start_positions = tf.reshape(start_positions, [-1])
            start_index = tf.one_hot(start_positions, depth=seq_len, axis=-1,
                                     dtype=tf.float32)
            start_features = tf.einsum(
                'lbh,bl->bh', sequence_output, start_index)
            start_features = tf.tile(start_features[None], [seq_len, 1, 1])
            end_logits = self.end_logits_proj_layer0(
                tf.concat([sequence_output, start_features], axis=-1))

            end_logits = self.end_logits_layer_norm(end_logits)

            end_logits = self.end_logits_proj_layer1(end_logits)
            end_logits = tf.transpose(tf.squeeze(end_logits, -1), [1, 0])
            end_logits_masked = end_logits * (1 - p_mask) - 1e30 * p_mask
            end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
        else:
            start_top_log_probs, start_top_index = tf.nn.top_k(
                start_log_probs, k=self.start_n_top)
            start_index = tf.one_hot(
                start_top_index, depth=seq_len, axis=-1, dtype=tf.float32)
            start_features = tf.einsum(
                'lbh,bkl->bkh', sequence_output, start_index)
            end_input = tf.tile(sequence_output[:, :, None], [
                                1, 1, self.start_n_top, 1])
            start_features = tf.tile(start_features[None], [seq_len, 1, 1, 1])
            end_input = tf.concat([end_input, start_features], axis=-1)
            end_logits = self.end_logits_proj_layer0(end_input)
            end_logits = tf.reshape(end_logits, [seq_len, -1, self.hidden_size])
            end_logits = self.end_logits_layer_norm(end_logits)

            end_logits = tf.reshape(end_logits,
                                    [seq_len, -1, self.start_n_top, self.hidden_size])

            end_logits = self.end_logits_proj_layer1(end_logits)
            end_logits = tf.reshape(
                end_logits, [seq_len, -1, self.start_n_top])
            end_logits = tf.transpose(end_logits, [1, 2, 0])
            end_logits_masked = end_logits * (
                1 - p_mask[:, None]) - 1e30 * p_mask[:, None]
            end_log_probs = tf.nn.log_softmax(end_logits_masked, -1)
            end_top_log_probs, end_top_index = tf.nn.top_k(
                end_log_probs, k=self.end_n_top)
            end_top_log_probs = tf.reshape(end_top_log_probs,
                                           [-1, self.start_n_top * self.end_n_top])
            end_top_index = tf.reshape(end_top_index,
                                       [-1, self.start_n_top * self.end_n_top])

        # an additional layer to predict answerability

        # get the representation of CLS
        cls_index = tf.one_hot(cls_index, seq_len, axis=-1, dtype=tf.float32)
        cls_feature = tf.einsum('lbh,bl->bh', sequence_output, cls_index)

        # get the representation of START
        start_p = tf.nn.softmax(start_logits_masked,
                                axis=-1, name='softmax_start')
        start_feature = tf.einsum('lbh,bl->bh', sequence_output, start_p)

        ans_feature = tf.concat([start_feature, cls_feature], -1)
        ans_feature = self.answer_class_proj_layer0(ans_feature)
        ans_feature = self.ans_feature_dropout(
            ans_feature, training=kwargs.get('training', False))
        cls_logits = self.answer_class_proj_layer1(ans_feature)
        cls_logits = tf.squeeze(cls_logits, -1)

        if kwargs.get("training", False):
            return (start_log_probs, end_log_probs, cls_logits)
        else:
            return (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits)


class ALBertQAModel(tf.keras.Model):

    def __init__(self, albert_config, max_seq_length, init_checkpoint, start_n_top, end_n_top, dropout=0.1, **kwargs):
        super(ALBertQAModel, self).__init__(**kwargs)
        self.albert_config = copy.deepcopy(albert_config)
        self.initializer = tf.keras.initializers.TruncatedNormal(
            stddev=self.albert_config.initializer_range)
        float_type = tf.float32

        input_word_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.layers.Input(
            shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')

        albert_layer = AlbertModel(config=albert_config, float_type=float_type)

        _, sequence_output = albert_layer(
            input_word_ids, input_mask, input_type_ids)

        self.albert_model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids],
                                           outputs=[sequence_output])
        if init_checkpoint != None:
            self.albert_model.load_weights(init_checkpoint)

        self.qalayer = ALBertQALayer(self.albert_config.hidden_size, start_n_top, end_n_top,
                                     self.initializer, dropout)

    def call(self, inputs, **kwargs):
        # unpacked_inputs = tf_utils.unpack_inputs(inputs)
        unique_ids = inputs["unique_ids"]
        input_word_ids = inputs["input_ids"]
        input_mask = inputs["input_mask"]
        segment_ids = inputs["segment_ids"]
        cls_index = tf.reshape(inputs["cls_index"], [-1])
        p_mask = inputs["p_mask"]
        if kwargs.get('training',False):
            start_positions = inputs["start_positions"]
        else:
            start_positions = None
        sequence_output = self.albert_model(
            [input_word_ids, input_mask, segment_ids], **kwargs)
        output = self.qalayer(
            sequence_output, p_mask, cls_index, start_positions, **kwargs)
        return (unique_ids,) + output

def get_raw_results_v2(predictions):
    """Converts multi-replica predictions to RawResult."""
    for unique_ids, start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits in zip(predictions['unique_ids'],
                                                    predictions['start_top_log_probs'],
                                                    predictions['start_top_index'],
                                                    predictions['end_top_log_probs'],
                                                    predictions['end_top_index'],
                                                    predictions['cls_logits']):
        for values in zip(unique_ids.numpy(), start_top_log_probs.numpy(), start_top_index.numpy(), end_top_log_probs.numpy(), end_top_index.numpy(), cls_logits.numpy()):
            yield squad_lib.RawResultV2(
                unique_id=values[0],
                start_top_log_probs=values[1].tolist(),
                start_top_index=values[2].tolist(),
                end_top_log_probs=values[3].tolist(),
                end_top_index=values[4].tolist(),
                cls_logits=values[5].tolist()
                )

def predict_squad_customized(strategy, albert_config,
                             predict_tfrecord_path, num_steps):
    """Make predictions using a Bert-based squad model."""
    if FLAGS.version_2_with_negative:
        predict_dataset = input_pipeline.create_squad_dataset_v2(
            predict_tfrecord_path,
            FLAGS.max_seq_length,
            FLAGS.predict_batch_size,
            is_training=False)
    else:
        pass

    predict_iterator = iter(
        strategy.experimental_distribute_dataset(predict_dataset))

    with strategy.scope():
        # add comments for #None,0.1,1,0
        if FLAGS.version_2_with_negative:
            squad_model = get_model_v2(albert_config, FLAGS.max_seq_length,
                                       None, 0.1, FLAGS.start_n_top, FLAGS.end_n_top, 0.0, 1, 0)
        else:
            pass

    checkpoint_path = FLAGS.checkpoint_path #tf.train.latest_checkpoint(FLAGS.model_dir)
    logging.info('Restoring checkpoints from %s', checkpoint_path)
    checkpoint = tf.train.Checkpoint(model=squad_model)
    checkpoint.restore(checkpoint_path).expect_partial()

    @tf.function
    def predict_step(iterator):
        """Predicts on distributed devices."""

        def _replicated_step(inputs):
            """Replicated prediction calculation."""
            x, _ = inputs
            if FLAGS.version_2_with_negative:
                y = squad_model(x, training=False)
                unique_ids, start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = y
                return dict(unique_ids=unique_ids,
                    start_top_log_probs=start_top_log_probs,
                    start_top_index=start_top_index,
                    end_top_log_probs=end_top_log_probs,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits)
            else:
                pass

        outputs = strategy.experimental_run_v2(
            _replicated_step, args=(next(iterator),))
        return tf.nest.map_structure(strategy.experimental_local_results, outputs)

    all_results = []
    for _ in range(num_steps):
        predictions = predict_step(predict_iterator)
        if FLAGS.version_2_with_negative:
            get_raw_results = get_raw_results_v2
        for result in get_raw_results(predictions):
            all_results.append(result)
        if len(all_results) % 100 == 0:
            logging.info('Made predictions for %d records.', len(all_results))
    return all_results

def get_model_v2(albert_config, max_seq_length, init_checkpoint, learning_rate,
                 start_n_top, end_n_top, dropout, num_train_steps, num_warmup_steps):
    """Returns keras model"""

    squad_model = ALBertQAModel(
        albert_config, max_seq_length, init_checkpoint, start_n_top, end_n_top, dropout)

    return squad_model


def predict_squad(strategy):
    """Makes predictions for a squad dataset."""
    albert_config = AlbertConfig.from_json_file(FLAGS.albert_config_file)
    doc_stride = FLAGS.doc_stride
    max_query_length = FLAGS.max_query_length

    eval_examples = squad_lib.read_squad_examples(
        input_file=FLAGS.predict_file,
        is_training=False)

    tokenizer = tokenization.FullTokenizer(vocab_file=None,
                                           spm_model_file=FLAGS.spm_model_file, do_lower_case=FLAGS.do_lower_case)

    eval_writer = squad_lib.FeatureWriter(
        filename=os.path.join(FLAGS.model_dir, 'eval.tf_record'),
        is_training=False)
    eval_features = []

    def _append_feature(feature):
        eval_features.append(feature)
        eval_writer.process_feature(feature)

    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on.
    dataset_size = squad_lib.convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        output_fn=_append_feature)
    eval_writer.close()

    logging.info('***** Running predictions *****')
    logging.info('  Num orig examples = %d', len(eval_examples))
    logging.info('  Num split examples = %d', len(eval_features))
    logging.info('  Batch size = %d', FLAGS.predict_batch_size)

    num_steps = math.ceil(dataset_size / FLAGS.predict_batch_size)
    all_results = predict_squad_customized(strategy, albert_config,
                                           eval_writer.filename, num_steps)

    output_prediction_file = os.path.join(FLAGS.model_dir, 'predictions.json')
    output_nbest_file = os.path.join(FLAGS.model_dir, 'nbest_predictions.json')
    output_null_log_odds_file = os.path.join(FLAGS.model_dir, 'null_odds.json')

    if FLAGS.version_2_with_negative:
        squad_lib.write_predictions_v2(
            eval_examples,
            eval_features,
            all_results,
            FLAGS.n_best_size,
            FLAGS.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            FLAGS.start_n_top,
            FLAGS.end_n_top
        )
    else:
        pass


def main(_):

    assert tf.version.VERSION.startswith('2.')
    logging.set_verbosity(logging.INFO)

    strategy = None
    if FLAGS.strategy_type == 'mirror':
        strategy = tf.distribute.MirroredStrategy()
    elif FLAGS.strategy_type == 'one':
        strategy = tf.distribute.OneDeviceStrategy('GPU:0')
    else:
        raise ValueError('The distribution strategy type is not supported: %s' %
                         FLAGS.strategy_type)
    if FLAGS.mode == 'predict':
        predict_squad(strategy)


if __name__ == '__main__':
    flags.mark_flag_as_required('albert_config_file')
    flags.mark_flag_as_required('model_dir')
    app.run(main)