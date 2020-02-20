from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import tensorflow as tf

from absl import app, flags, logging

flags.DEFINE_string('input_nbest_files',None,'')
flags.DEFINE_string('input_null_files',None,'')
flags.DEFINE_float('null_score_thresh',None,'')
flags.DEFINE_string('output_file',None,'')


FLAGS = flags.FLAGS

def main(_):

  if not FLAGS.input_null_files:
    raise ValueError("No input null files specified")

  all_scores = collections.OrderedDict()
  for input_file in FLAGS.input_null_files.split(","):
    with open(input_file, "r") as reader:
      input_data = json.load(reader)
      for (key, score) in input_data.items():
        if key not in all_scores:
          all_scores[key] = []
        all_scores[key].append(score)

  output_scores = {}
  for (key, scores) in all_scores.items():
    mean_score = 0.0
    for score in scores:
      mean_score += score
    mean_score /= float(len(scores))
    output_scores[key] = mean_score

  if not FLAGS.input_nbest_files:
    raise ValueError("No input nbest files specified")

  all_nbest = collections.OrderedDict()
  for input_file in FLAGS.input_nbest_files.split(","):
    with open(input_file, "r") as reader:
      input_data = json.load(reader)
      for (key, entries) in input_data.items():
        if key not in all_nbest:
          all_nbest[key] = collections.defaultdict(float)
        for entry in entries:
          all_nbest[key][entry["text"]] += entry["probability"]

  output_predictions = {}
  for (key, entry_map) in all_nbest.items():
    null_score = output_scores[key]
    if null_score >= FLAGS.null_score_thresh:
      output_predictions[key] = ""
    else:
      sorted_texts = sorted(
          entry_map.keys(), key=lambda x: entry_map[x], reverse=True)
      best_text = sorted_texts[0]
      output_predictions[key] = best_text

  with open(FLAGS.output_file, "w") as writer:
    writer.write(
        json.dumps(output_predictions, sort_keys=True, indent=4) + "\n")

if __name__ == '__main__':
  flags.mark_flag_as_required('input_nbest_files')
  flags.mark_flag_as_required('input_null_files')
  flags.mark_flag_as_required('null_score_thresh')
  flags.mark_flag_as_required('output_file')
  app.run(main)