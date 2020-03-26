import tensorflow as tf
import numpy as np

'''
Description: Script which contains the model definition and call functions
'''

class EncoderModel(tf.keras.Model):
  def __init__(self, vocab_size, dropout_ratio=0.5, embedding_dim=512):
    super(EncoderModel, self).__init__()
    self.keep_ratio = 1 - dropout_ratio
    self.encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.enc_lstm = tf.keras.layers.GRU(embedding_dim, return_state=True)

  def call(self, input_seq, training):
    """
    Description: Run the forward pass of EncoderModel.
    """
    y = self.encoder_embedding(input_seq)
    if training:
      y = tf.nn.dropout(y, self.keep_ratio)
    outputs, states = self.enc_lstm(y, training=training)
    return states


class DecoderModel(tf.keras.Model):
  def __init__(self, vocab_size, bias_init, embedding_dim=512, dropout_ratio=0.5):
    super(DecoderModel, self).__init__()
    self.keep_ratio = 1 - dropout_ratio
    self.embedding_dim = embedding_dim
    self.decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.dec_lstm = tf.keras.layers.GRU(embedding_dim, return_sequences=True)
    self.dec_dense = tf.keras.layers.Dense(vocab_size, None, True, 'glorot_uniform', bias_init)

  def call(self, input_seq, initial_state, training):
    """
    Description: Run the forward pass of DecoderModel.
    """
    y = self.decoder_embedding(input_seq)
    if training:
      y = tf.nn.dropout(y, self.keep_ratio)
    outputs =  self.dec_lstm(y, initial_state=initial_state, training=training) 
    reshaped_out = tf.reshape(outputs, [-1, self.embedding_dim])
    linear_out = self.dec_dense(reshaped_out)
    return linear_out


class Seq2SeqModel(tf.keras.Model):
  def __init__(self, vocab_size, bias_init, embedding_dim=512, dropout_rate=0.5):
    super(Seq2SeqModel, self).__init__()    
    self.encoder = EncoderModel(vocab_size)
    self.decoder = DecoderModel(vocab_size, bias_init)

  def call(self, input_seq, targets, training):
    """
    Description: Run the forward pass of Seq2SeqModel.
    """        
    enc_state = self.encoder(input_seq, training)  
    #dec_input_1 = (batch_size, 1)- since we assume the first input to decoder to be the last entry of input to encoder
    dec_input_1 = tf.expand_dims(input_seq[:, -1], 1)
 
    dec_input_2 = targets[:,:tf.shape(targets)[1]-1]
    dec_input = tf.concat([dec_input_1, dec_input_2], axis =1) 
    dec_state = enc_state
    
    dec_outputs = self.decoder(dec_input, dec_state, training)
    return dec_outputs, enc_state

 


     

