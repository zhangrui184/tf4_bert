import tensorflow as tf

import os
import numpy as np
#import time
#import re
#import jieba
# 超参数
# Number of Epochs
epochs =5

# Batch Size
batch_size =5
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 128
decoding_embedding_size = 128
# Learning Rate
learning_rate = 0.05
vocab_path="D:\python project me\data/text_summar/nice data/vocab.txt"
def get_inputs():
    '''
    模型输入tensor
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length
def get_vocab(vocab_path):
    vocab_list=[]
    with open(vocab_path,'r',encoding='utf-8')as f:
        for item in f.readlines():
            vocab_list.append(item.strip())
    int_to_vocab = {idx: word for idx, word in enumerate(vocab_list)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab,vocab_to_int
source_int_to_letter, source_letter_to_int=get_vocab(vocab_path)
target_int_to_letter, target_letter_to_int =source_int_to_letter, source_letter_to_int

def get_encoder_layer(input_data, rnn_size, num_layers,
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):

    '''
    构造Encoder层
    
    参数说明：
    - input_data: 输入tensor
    - rnn_size: rnn隐层结点数量
    - num_layers: 堆叠的rnn cell数量
    - source_sequence_length: 源数据的序列长度
    - source_vocab_size: 源数据的词典大小
    - encoding_embedding_size: embedding的大小
    '''
    # Encoder embedding
    encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    # RNN cell
    def get_lstm_cell(rnn_size):
        lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return lstm_cell

    cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
    
    encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input, 
                                                      sequence_length=source_sequence_length, dtype=tf.float32)
    
    return encoder_output, encoder_state
def process_decoder_input(data, vocab_to_int, batch_size):
    '''
    补充<GO>，并移除最后一个字符
    
    这里是为了构建 Decoder训练时的输入数据，使用target 而不是预测出的数据，提高精度
    '''
    # cut掉最后一个字符
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['_GO']), ending], 1)

    return decoder_input
def decoding_layer(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, encoder_state, decoder_input,
                   encoder_outputs,source_sequence_length):
    '''
    构造Decoder层
    
    参数：
    - target_letter_to_int: target数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    
    - encoder_outputs :添加一个注意力机制
    - source_sequence_length 源数据长度
    '''
    
    '''
    # attention_states: [batch_size, max_time, num_units]
    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

    # Create an attention mechanism
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, attention_states,
        memory_sequence_length=source_sequence_length)
    
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    decoder_cell, attention_mechanism,
    attention_layer_size=num_units)
    
    
    '''
    # 1. Embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

    # 2. 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return decoder_cell
    
    #2.1 添加注意力机制的RNN 单元
    def get_decoder_cell_attention(rnn_size):
         # attention_states: [batch_size, max_time, num_units]
#        attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
        attention_states=encoder_outputs
         # Create an attention mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        rnn_size, attention_states,
        memory_sequence_length=source_sequence_length)
        
        decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                            decoder_cell, attention_mechanism,
                            attention_layer_size=rnn_size)
        
        return decoder_cell
    
    
#     cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])
    cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell_attention(rnn_size) for _ in range(num_layers)])



    
    # 3. Output全连接层
    output_layer = tf.layers.Dense(target_vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))


    # 4. Training decoder
    with tf.variable_scope("decode"):
        # 得到help对象
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)
        # 构造decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                           training_helper,
                                                           initial_state=cell.zero_state(dtype=tf.float32,batch_size=batch_size)
                                                        ,output_layer=output_layer) 

        training_decoder_output, _ ,_= tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                       impute_finished=True,
                                                                       maximum_iterations=max_target_sequence_length)
    # 5. Predicting decoder
    # 与training共享参数
    with tf.variable_scope("decode", reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([target_letter_to_int['_GO']], dtype=tf.int32), [batch_size],
                               name='start_tokens')
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                start_tokens,
                                                                target_letter_to_int['_EOS'])
        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                        predicting_helper,
                                                        initial_state=cell.zero_state(dtype=tf.float32,batch_size=batch_size)
                                                            ,output_layer=output_layer)
        predicting_decoder_output, _,_  = tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                            impute_finished=True,
                                                            maximum_iterations=max_target_sequence_length)
    
    return training_decoder_output, predicting_decoder_output
def seq2seq_model(input_data, targets, lr, target_sequence_length, 
                  max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size,
                  encoder_embedding_size, decoder_embedding_size, 
                  rnn_size, num_layers):
    
    # 获取encoder的状态输出
    encoder_outputs, encoder_state = get_encoder_layer(input_data, 
                                  rnn_size, 
                                  num_layers, 
                                  source_sequence_length,
                                  source_vocab_size, 
                                  encoding_embedding_size)
    
    
    # 预处理后的decoder输入
    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)
    
    # 将状态向量与输入传递给decoder
    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int, 
                                                                       decoding_embedding_size, 
                                                                       num_layers, 
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       encoder_state, 
                                                                       decoder_input,
                                                                        encoder_outputs,
                                                                        source_sequence_length
                                                                       ) 
    
    return training_decoder_output, predicting_decoder_output

#----------------------------------------------------------------------------------------------------------------

# 构造graph
train_graph = tf.Graph()

with train_graph.as_default():
    
    # 获得模型输入    
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
    
    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data, 
                                                                      targets, 
                                                                      lr, 
                                                                      target_sequence_length, 
                                                                      max_target_sequence_length, 
                                                                      source_sequence_length,
                                                                      len(source_letter_to_int),
                                                                      len(target_letter_to_int),
                                                                      encoding_embedding_size, 
                                                                      decoding_embedding_size, 
                                                                      rnn_size, 
                                                                      num_layers)    
    
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
    
    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
#------------------------------------------------------------------------------------------------------
def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    
    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]
def get_batches(file_list,tokenize_path,batch_size,pad_int):
    '''
    定义生成器，用来获取tokenize下的所有content
    '''
    #for item in file_list:
        #source_path=os.path.join(tokenize_path,'content_'+item)
    source_path="D:\python project me\data/text_summar/nice data/nice data train\content_train_id.txt"
       # target_path=os.path.join(tokenize_path,'title_'+item)
    target_path="D:\python project me\data/text_summar/nice data/nice data train/title_train_id.txt"
    with open(source_path,'r',encoding='utf-8')as sf:
        sources=[[int(word) for word in sentence.strip().split(' ')]for sentence in sf.readlines()]
    with open(target_path,'r',encoding='utf-8')as tf:
        targets=[[int(word) for word in sentence.strip().split(' ')]for sentence in tf.readlines()]
        
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

            #yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths
        return pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

#-----------------------------------------------------------------------------------------------------------------------------------

data_path='D:\python project me\data/text_summar/nice data/nice data train'
file_list=os.listdir(data_path)
tokenize_path='D:\python project me\data/text_summar/nice data/tokenize'
batch_size=5
pad_int=source_letter_to_int['_PAD']
#-------------------------------------------------------------------------------------------------------------------------------------
#valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths = next(get_batches(file_list,tokenize_path,batch_size,pad_int))
valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths = get_batches(file_list,tokenize_path,batch_size,pad_int)
display_step = 50 # 每隔50轮输出loss


# 查看是否有 checkpoint_f 这个文件，无则新建一个
if not os.path.exists('./checkpoint_f'):
        os.makedirs('./checkpoint_f')

checkpoint = "./checkpoint_f/trained_model.ckpt" 
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
        
    for epoch_i in range(1, epochs+1):
        #for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
               # get_batches(file_list,tokenize_path,batch_size,pad_int)):
        for batch_i in range(batch_size):
            targets_batch, sources_batch, targets_lengths, sources_lengths = valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths
            for i in range(100):

               _, loss = sess.run(
                 [train_op, cost],
                {input_data: sources_batch,
                 targets: targets_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths})

           # if batch_i % display_step == 0:
            
                # 计算validation loss
               validation_loss = sess.run(
                [cost],
                {input_data: valid_sources_batch,
                 targets: valid_targets_batch,
                 lr: learning_rate,
                 target_sequence_length: valid_targets_lengths,
                 source_sequence_length: valid_sources_lengths})
                
               print('i {:} Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(i,epoch_i,
                              epochs, 
                              batch_i, 
                              '未知', 
                              loss, 
                              validation_loss[0]))

    
    
    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print('Model Trained and Saved')
#---------------------------------------------------------------------------------------------------------------------------------------
