import tensorflow as tf
import numpy as np

class EncoderRNN(tf.keras.layers.Layer):
    def __init__(self,num_units,num_layers,dropout):
        super(EncoderRNN,self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units = self.num_units,return_sequences = True,return_state = True,dropout=self.dropout),name='layer'+str(i)))
    def call(self,inputs,is_training):
        outs = self.layers[0](inputs=inputs,training = is_training)
        inputs = outs[0]
        for i in range(1,self.num_layers):
            outs = self.layers[i](inputs,training = is_training)
            inputs = outs[0]
        outputs = outs[0]
        states = outs[1:]
        return outputs,states


#print(encoder.trainable_variables)

#attention layer
def scaleddotproduct(q,k,v,mask=None):
  att_wts = tf.matmul(q,k,transpose_b=True)
  dk = tf.cast(tf.shape(k)[-1],dtype=tf.float32)
  att_wts = att_wts/(tf.math.sqrt(dk))
  if (mask!=None):
    att_wts += mask*(-1e9)
  att_wts = tf.nn.softmax(att_wts,axis=-1)
  ctx_vec = tf.matmul(att_wts,v)
  return ctx_vec,att_wts

class Multihead_Att(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads):
        super(Multihead_Att,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model%num_heads == 0
        
        self.depth = d_model//num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
    def split_heads(self,x,batch_size):
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(x,perm = [0,2,1,3])
    def call(self,q,v,k,mask=None):
        q = self.wq(q)
        v = self.wv(v)
        k = self.wk(k)
        batch_size = tf.shape(q)[0]
        print(batch_size)
        q = self.split_heads(q,batch_size)
        v = self.split_heads(v,batch_size)
        k = self.split_heads(k,batch_size)
        
        att_out,att_weights = scaleddotproduct(q,k,v,mask)
        att_out = tf.transpose(att_out,perm =[0,2,1,3] )
        att_out = tf.reshape(att_out,(batch_size,-1,self.d_model))
        output = self.dense(att_out)
        
        return output,att_weights

class DecoderRNN(tf.keras.layers.Layer):
    def __init__(self,num_layers,num_units,dropout,d_model,num_heads):
        super(DecoderRNN,self).__init__()
        self.num_layers = num_layers
        self.num_units = num_units
        self.dropout = dropout
        self.d_model = d_model
        self.num_heads = num_heads
        self.layers = [tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units = self.num_units,dropout = self.dropout,return_sequences = True,return_state = True,name='decoder_layer'+str(i+1))) for i in range(num_layers)]
        self.output_layer = tf.keras.layers.Dense(units = 1025)
        self.stop_layer = tf.keras.layers.Dense(units =1 ,activation = tf.nn.sigmoid)
        self.attention = Multihead_Att(d_model=self.d_model,num_heads=self.num_heads)

    def call(self,inputs,encoder_outs,is_training):
        ctx_vec ,att_wts = self.attention(inputs,encoder_outs,encoder_outs)
        inputs = tf.concat([inputs,ctx_vec],axis=-1)
        for i in range(self.num_layers):
            outs = self.layers[i](inputs=inputs,training=is_training)
            inputs = outs[0]
        outputs = self.output_layer(inputs)
        stop_token = self.stop_layer(inputs)
        return outputs,stop_token,outs



#print(len(d.trainable_variables))

optimizer = tf.keras.optimizers.Adam()
loss_object1 = tf.keras.losses.MeanSquaredError()
loss_object2 = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def mse_loss(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  print(mask)
  loss_ = loss_object1(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


def crossentropy_loss(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object2(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

def stop_targ(sequence_lengths):
  batch_size = len(sequence_lengths)
  stop_mat = []
  max_len = max(sequence_lengths)
  for i in range(batch_size):
    stop_arr = []
    for j in range(max_len):
      if (j == sequence_lengths[i]-1):
        stop_arr.append(1)
        continue
      stop_arr.append(0)
    stop_mat.append(stop_arr)
  stop_mat = tf.convert_to_tensor(stop_mat,dtype = tf.float32)
  return stop_mat	

@tf.function
def train_step(src_inp,targ_inp,stop_inp):
	spec_loss=0
	stop_loss=0
	with tf.GradientTape() as tape:
		encoder_outputs,states = encoder(src_inp,True)
		batch_size = targ_inp.shape[0]
		seq_len = targ_inp.shape[1]
		enc_units = encoder_outputs.shape[-1]
		decoder_input = tf.zeros(shape=(batch_size,1,enc_units))
		for i in range(seq_len):
			spec_outputs,stop_outs,decoder_outs = decoder(decoder_input,encoder_outputs,True)
			print("shape of stop tokens",stop_outs.shape)
			print("time step---",i)
			stop_outs = tf.reshape(stop_outs,shape=(batch_size,))
			spec_loss += mse_loss(targ_inp[:,i,:],spec_outputs)
			stop_loss += crossentropy_loss(stop_inp[:,i],stop_outs)
			decoder_input = decoder_outs[0]
			print(decoder_input.shape)
	spec_loss /= seq_len
	stop_loss /= seq_len
	loss = tf.reduce_mean(spec_loss+stop_loss)
	variables = encoder.trainable_variables+decoder.trainable_variables
	gradients = tape.gradient(loss,variables)
	optimizer.apply_gradients(zip(gradients,variables))
	return loss

encoder = EncoderRNN(1024,4,0.1)

decoder = DecoderRNN(2,1024,0.1,1024,4)

inputs = tf.random.uniform(shape=(16,128,1024))

targs = tf.random.uniform(shape=(16,128,1025))

stop_targets = stop_targ([128 for i in range(16)])
k = train_step(inputs,targs,stop_targets)
print(k)



