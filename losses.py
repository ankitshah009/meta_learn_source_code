import numpy as np
import tensorflow as tf
from model import Seq2SeqModel

'''
Description: Script which contains all loss functions
'''

def hole_loss(model_new, hole_window, hole_target, seq_len_hole_target, training):
  """
  Description: Calculates hole target loss given hole window and model instance
  """
  labels = tf.reshape(hole_target, [-1])
  batch_size = hole_target.shape[0]
  max_targ_len_hole = hole_target.shape[1]
  #outputs = (max_target_time_steps*batch_size, vocab_size)
  outputs, states = model_new(hole_window, hole_target, training)
  #loss = (max_target_time_steps*batch_size)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs)
  #reshaped_loss = (batch_size, max_target_time_steps)
  reshaped_loss = tf.reshape(loss, [batch_size, max_targ_len_hole])
  # mask = (max_target_time_steps, batch_size)
  mask = tf.sequence_mask(seq_len_hole_target, max_targ_len_hole, dtype=tf.float32)
  #masked_loss = (batch_size, max_target_time_steps)
  masked_loss = tf.multiply(reshaped_loss, mask)
  #batch_loss = real number = avg loss sequences in the batch [This is the token loss]
  batch_loss = tf.reduce_sum(masked_loss)/ tf.cast(len(seq_len_hole_target), dtype=tf.float32)
  return batch_loss, masked_loss

def clip_gradients(grads_and_vars, clip_ratio=0.25):
  """
  Description: Performs gradient clipping
  """
  gradients, variables = zip(*grads_and_vars)
  clipped, _ = tf.clip_by_global_norm(gradients, clip_ratio)
  return zip(clipped, variables)

def inner_loss_eval(model, sup_window, sup_target, seq_len_sup_target, training, method, inner_learning_rate, batch_size_sup, num_of_updates):
  """
  Description: Calculates the inner loss during evaluation. Returns the updated model instance
  """
  if method=='dyn_eval':
    model_new = dyn_eval_loss(model, sup_window, sup_target, seq_len_sup_target, training, inner_learning_rate)
    return model_new

  if method=='tssa':
    model_new = support_loss_eval(model, sup_window, sup_target, seq_len_sup_target, training, inner_learning_rate, batch_size_sup, num_of_updates)
    return model_new


def one_batch_support_loss(model, inputs, targets, target_seq_len, training):
  """
  Description: Calculates support token loss given support window over one batch
  """
  batch_size = targets.shape[0]
  max_targ_len = targets.shape[1]
  outputs, enc_sup_state = model(inputs, targets, training)
  labels = tf.reshape(targets, [-1])
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs)
  reshaped_loss = tf.reshape(loss, [batch_size, max_targ_len])
  mask = tf.sequence_mask(target_seq_len, max_targ_len, dtype=tf.float32)
  masked_loss = tf.multiply(reshaped_loss, mask)
  batch_loss = tf.reduce_sum(masked_loss)/ tf.cast(len(target_seq_len), dtype=tf.float32)
  return batch_loss, None

def support_loss_eval(model, inputs, targets, target_seq_len, training, inner_learning_rate, batch_size_sup, num_of_updates):
  """
  Description: Calculates support token loss given support window for num_of_updates updates over all batches during evaluation
  """
  batch_size = targets.shape[0]
  max_targ_len = targets.shape[1]

  # repeated till exhaustion of number of updates
  dataset_sup = tf.data.Dataset.from_tensor_slices((inputs, targets, target_seq_len)).repeat().shuffle(batch_size)
  dataset_sup = dataset_sup.batch(batch_size_sup, drop_remainder=False)

  # to reset optimizer state
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = inner_learning_rate)

  for (batch, (inputs, targets, target_seq_len)) in enumerate(dataset_sup):
    if batch >= num_of_updates:
      break
    with tf.GradientTape() as g:
      loss, attention_weights = one_batch_support_loss(model, inputs, targets, target_seq_len, training)
    grads = g.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(clip_gradients(zip(grads, model.trainable_variables)))

  return model

def support_loss_train(model, inputs, targets, target_seq_len, training, optimizer, batch_size_sup, num_of_updates):
  """
  Description: Calculates support token loss given support window for num_of_updates updates over all batches during training
  """
  batch_size = targets.shape[0]
  max_targ_len = targets.shape[1]

  # repeated till exhaustion of number of updates
  dataset_sup = tf.data.Dataset.from_tensor_slices((inputs, targets, target_seq_len)).repeat().shuffle(batch_size)
  dataset_sup = dataset_sup.batch(batch_size_sup, drop_remainder=False)

  for (batch, (inputs, targets, target_seq_len)) in enumerate(dataset_sup):
    if batch >= num_of_updates:
      break
    with tf.GradientTape() as g:
      loss, attention_weights = one_batch_support_loss(model, inputs, targets, target_seq_len, training)
    grads = g.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(clip_gradients(zip(grads, model.trainable_variables)))

  return model

def dyn_eval_loss(model, inputs, targets, target_seq_len, training, inner_learning_rate):
  """
  Description: Calculates dynamic evluation loss
  """
  # batch-size = 20 to compare with baseline
  dataset_sup = tf.data.Dataset.from_tensor_slices((inputs, targets, target_seq_len)).batch(20)
  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = inner_learning_rate)
  for (batch, (inputs, targets, target_seq_len)) in enumerate(dataset_sup):
      with tf.GradientTape() as g:
        loss, attention_weights = one_batch_support_loss(model, inputs, targets, target_seq_len, training)
      grads = g.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(clip_gradients(zip(grads, model.trainable_variables)))

  return model