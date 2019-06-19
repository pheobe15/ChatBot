import numpy as np
import tensorflow as tf
import re
import time

#DATA PREPROCESSING
lines= open('movie_lines.txt', encoding='utf-8', errors= 'ignore').read().split('\n')
conversations= open('movie_conversations.txt', encoding='utf-8', errors= 'ignore').read().split('\n')

id2line={}
for line in lines:
    _line=line.split(' +++$+++ ')
    if len(_line)==5:
        id2line[_line[0]]= _line[4]

conversations_ids=[]
for conversation in conversations[:-1]:
    _conversation=conversation.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conversation.split(','))


questions=[]
answers=[]
for conversation in conversations_ids:
    for i in range(len(conversation) -1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
def cleantext(text):
    text=text.lower()
    text=re.sub(r"i'm", "i am" ,text)
    text=re.sub(r"he's", "he is" ,text)
    text=re.sub(r"she's", "she is" ,text)
    text=re.sub(r"that's", "that is" ,text)
    text=re.sub(r"what's", "what is" ,text)
    text=re.sub(r"where's", "where is" ,text)
    text=re.sub(r"\'ll", " will" ,text)
    text=re.sub(r"\'ve", " have" ,text)
    text=re.sub(r"\'re", " are" ,text)
    text=re.sub(r"\'d", " would" ,text)
    text=re.sub(r"won't", "will not" ,text)
    text=re.sub(r"can't", "cannot" ,text)
    text=re.sub(r"[!@#$%^&*_+-=\"\'?><,;]", "" ,text)
    return text                
    
    
clean_questions=[]
for question in questions:
    clean_questions.append(cleantext(question))

clean_answers=[]
for answer in answers:
    clean_answers.append(cleantext(answer))

#creating dict to map num of occurences so we can remove rarely used words
word2count={}
for question in questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
for answer in answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1

            
threshold=20
questionsword2int={} 
word_count=0
for word,count in word2count.items():
    if count >= threshold:
        questionsword2int[word]= word_count
        word_count +=1
answersword2int={} 
word_count=0
for word,count in word2count.items():
    if count >= threshold:
        answersword2int[word]= word_count
        word_count +=1
               
tokens=["<PAD>","<EOS>","<OUT>","<SOS>"]
for token in tokens:
    questionsword2int[token]= len(questionsword2int) +1
for token in tokens:
    answersword2int[token]= len(answersword2int) +1
    

answersint2word= {w_i:w for w,w_i in answersword2int.items() }

for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"
    
#translating all the words into integers
#and filter all the least frequently used words by <OUT>
question_into_int=[]
for question in clean_questions:
    ints=[]
    for word in question.split():
        if word not in questionsword2int:
            ints.append(questionsword2int["<OUT>"])
        else:
            ints.append(questionsword2int[word])
    question_into_int.append(ints)        
    
answer_into_int=[]
for answer in clean_answers:
    ints=[]
    for word in answer.split():
        if word not in answersword2int:
            ints.append(answersword2int["<OUT>"])
        else:
            ints.append(answersword2int[word])
    answer_into_int.append(ints)  

      
#sorting ques and ans by length as it will speed and optimise training by reducing the length of padding
sorted_clean_questions=[]
sorted_clean_answers=[]
for length in range(1, 25 + 1):
   #adding +1 coz upper bound in python in not included
   for i in enumerate(question_into_int):
       #print(i)
       #gives a couple of index num and the question
       if len(i[1]) == length:
           sorted_clean_questions.append(question_into_int[i[0]])
           sorted_clean_answers.append(answer_into_int[i[0]])
       
       
#Building Seq2seq model
           
def model_inputs():
    inputs= tf.placeholder(tf.int32, [None][None], name="input")
    targets= tf.placeholder(tf.int32, [None][None], name="target")       
    lr= tf.placeholder(tf.float32, name="learning_rate")   
    keep_prob= tf.placeholder(tf.float32, name="keep_prob")
    return inputs, targets, lr, keep_prob
       
#preprocessing placeholders because decoder accepts data only in batches
def preprocess_targets(targets, word2int, batchsize):
    left_side= tf.fill([batchsize,1], word2int['<SOS>'])
    right_side= tf.strided_slice(targets, [0,0], [batchsize, -1], [1,1])
    preprocessed_targets= tf.concat([left_side, right_side], 1)
    return preprocessed_targets

def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm= tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout= tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob= keep_prob)
    encoder_cell= tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state= tf.nn.bidirectional_dynamic_rnn(cell_fw= encoder_cell,
                                                      cell_bw= encoder_cell,
                                                      sequence_length= sequence_length,
                                                      inputs= rnn_inputs,
                                                      dtype= tf.float32)
    return encoder_state

#decoding training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states= tf.zeroes([batchsize, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function= tf.contrib.seq2seq.prepare_attention(attention_states, attention_option= 'bahdanau', num_units= decoder_cell.output_size)
    training_decoder_function= tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                             attention_keys,
                                                                             attention_values,
                                                                             attention_score_function,
                                                                             attention_construct_function,
                                                                             name= 'attn_doc_train')
    decoder_output, decoder_final_state, decoder_final_context_state= tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                            training_decoder_function,
                                                                                                            decoder_embedded_input,
                                                                                                            sequence_length,
                                                                                                            scope= decoding_scope)
    decoder_output_dropout= tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

#decoding test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states= tf.zeroes([batchsize, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function= tf.contrib.seq2seq.prepare_attention(attention_states, attention_option= 'bahdanau', num_units= decoder_cell.output_size)
    test_decoder_function= tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                             encoder_state[0],
                                                                             attention_keys,
                                                                             attention_values,
                                                                             attention_score_function,
                                                                             attention_construct_function,
                                                                             decoder_embeddings_matrix,
                                                                             sos_id,
                                                                             eos_id,
                                                                             maximum_length,
                                                                             num_words,
                                                                             name= 'attn_doc_inf')    #inf=inference
    test_predictions, decoder_final_state, decoder_final_context_state= tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                               test_decoder_function,
                                                                                                               scope= decoding_scope)
    return test_predictions


    





































































       
    























    
    
    