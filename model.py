import codecs

import torch
import torch.nn as nn
import torch.nn.functional as F
from word_embeddings import WordEmbedder

INPUT_MAX_SEQ_LEN = 7
KEYWORD_MAX_SEQ_LEN = 3
word_embedder = WordEmbedder()
voc_size = 5292

#TODO change this to accept a sequence, now sequence alwasy =1
class EncoderRNN(nn.Module):

    def __init__(self, pretrained_embedding, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)

    # shape of input: (seq_len, batch)
    #shape of hidden/cell : num_layers * num_directions, batch, hidden_size
    def forward(self, input, hidden, cell):
        embedding = self.embedding(input)
        return self.lstm(embedding.float(), (hidden, cell))


class AttentionDecoderRNN(nn.Module):

    def __init__(self, hidden_size):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.attn1 = torch.nn.Linear(hidden_size*3, hidden_size)
        self.attn2 = torch.nn.Linear(hidden_size, 1)
        self.lstm = nn.LSTM(3*hidden_size, hidden_size, bidirectional=False) # input size is the same as output size of encoder LSTM . *2 due to bidirectional of encoder
        self.out = nn.Linear(hidden_size, voc_size)

    # shape(encode_outputs) -> (seq_len+1, batch,  hidden*2)
    #shape(mask) -> (seq_len+1, batch)  , here mask[i]=1 means i is not used
    # shape(hidden) -> (1, batch, hidden)
    def forward(self, hidden, cell, encode_outputs, mask):
        seq_len = encode_outputs.shape[0]-1
        repeat_hidden = hidden.repeat(seq_len+1,1, 1)
        energies = self.attn2(F.tanh(self.attn1( torch.cat((repeat_hidden, encode_outputs), dim=2)))).squeeze(2)

        energies[mask] = float('-inf')
        atten_weights = F.softmax(energies, dim=0)   #(seq_len+1,batch)
        attn_applied = torch.bmm(atten_weights.unsqueeze(dim=1).transpose(0,2) , encode_outputs.transpose(0,1)).view(-1, self.hidden_size*2)
        cur_input = torch.cat( (hidden,attn_applied.unsqueeze(dim=0)), dim=2)

        output, (hidden,cell) = self.lstm(cur_input, (hidden, cell))
        # print(output==hidden)
        # print(F.softmax(self.out(output), dim=2).max())
        return F.log_softmax(self.out(output), dim=2).squeeze(0), (hidden, cell)



def train():
    training_data = [('酒', '好花好月需好酒'), ('酒','好花好月需'),('酒好','好花好月需好酒'), ('酒','好花好月需好酒')]
    # training_data = read_training()
    print('training size is {}'.format(len(training_data)))
    batch_data = prepare_batch_training_data(training_data, batch_size=128)
    keyword_encoder = EncoderRNN(torch.from_numpy(word_embedder.get_word_embedding(512)) , 512).cuda()

    input_encoder = EncoderRNN(torch.from_numpy(word_embedder.get_word_embedding(512))  , 512).cuda()

    output_decoder = AttentionDecoderRNN(512).cuda()

    # keyword_encoder.load_state_dict(torch.load('./data/model/keyword.mdl', map_location=torch.device('cpu')))
    # input_encoder.load_state_dict(torch.load('./data/model/input_encoder.mdl', map_location=torch.device('cpu')))
    # output_decoder.load_state_dict(torch.load('./data/model/output_decoder.mdl', map_location=torch.device('cpu')))


    criterion = nn.NLLLoss()
    keyword_encoder_optimizer = torch.optim.Adam(keyword_encoder.parameters(), lr=1e-3)
    input_encoder_optimizer = torch.optim.Adam(input_encoder.parameters(), lr=1e-3)

    output_decoder_optimizer = torch.optim.Adam(output_decoder.parameters(), lr=1e-3)
    best_loss = 1000
    for itr in range(65000):
        print("iteration  " + str(itr))

        avg_loss_per_iter =0
        for idx, data in enumerate(batch_data):
           loss = 0
           keyword_encoder_optimizer.zero_grad()
           input_encoder_optimizer.zero_grad()
           output_decoder_optimizer.zero_grad()
           keyword, input_sentence, output_sentence, keyword_mask = data
           keyword = torch.tensor(keyword, dtype=torch.long).cuda()
           input_sentence = torch.tensor(input_sentence, dtype=torch.long).cuda()
           output_sentence = torch.tensor(output_sentence, dtype=torch.long).cuda()
           keyword_mask = torch.tensor(keyword_mask).cuda()
           # print(output_sentence.shape)
           keyword_decoder_hidden = torch.zeros(2, keyword.shape[0] , 512, dtype=torch.float).float().cuda()
           keyword_decoder_cell = torch.zeros(2,keyword.shape[0],512, dtype=torch.float).float().cuda()

           # print('zzzz', keyword.shape)
           keyword_decoder_outputs, (keyword_decoder_hidden, keyword_decoder_cell) = keyword_encoder(keyword.transpose(0,1), keyword_decoder_hidden ,keyword_decoder_cell)

           # print(keyword_decoder_outputs.shape)


           input_decoder_hidden = torch.zeros(2, input_sentence.shape[0], 512, dtype=torch.float).float().cuda()
           input_decoder_cell = torch.zeros(2, input_sentence.shape[0], 512, dtype=torch.float).float().cuda()
           # print(input_sentence)
           input_encoder_outputs, (input_decoder_hidden, input_decoder_cell) = input_encoder(input_sentence.transpose(0,1), input_decoder_hidden, input_decoder_cell)

           # print(keyword_decoder_outputs[0, : , 0:512].shape)
           # print(keyword_decoder_outputs.shape)
           # print(keyword)
           # print('asd')


           a = keyword_mask.unsqueeze(dim=2).repeat(1,1, 1024).transpose(0,1)

           # print(torch.cat((keyword_decoder_outputs[0, :, 0:512], keyword_decoder_outputs[0, :, 512:]),dim=1))

           h0 = torch.cat((keyword_decoder_outputs[0, : , 0:512], keyword_decoder_outputs[a].reshape(-1, 1024)[:, 512:]), dim=1)

           # print(h0.shape)
           # print(h0.shape)
           encoder_outputs = torch.cat( ( h0.unsqueeze(0), input_encoder_outputs), dim=0)
           print(encoder_outputs[:,0,:])

           encoder_mask = torch.cat( ( torch.zeros(input_sentence.shape[0], 1, dtype=torch.long).cuda() , input_sentence ), dim=1)==5291


           hidden = torch.zeros(1, keyword.shape[0] , 512, dtype=torch.float).float().cuda()
           cell = torch.zeros(1, keyword.shape[0] , 512, dtype=torch.float).float().cuda()

           decoder_outputs = []

           # output_sentences =
           # [seqlen, batch]

           output_mask = (output_sentence!=5291).transpose(0,1)
           for i in range(7):
               decoder_output, (hidden, cell) = output_decoder(hidden, cell, encoder_outputs, encoder_mask.transpose(0,1))

               # print("aa", decoder_output)
               # print(output_sentence[:,i])
               # print(output_sentence)
               # print(output_mask[i])
               if any(output_mask[i]):
                loss += criterion(decoder_output[output_mask[i]], output_sentence[:,i][output_mask[i]])
           loss.backward()
           keyword_encoder_optimizer.step()
           input_encoder_optimizer.step()
           output_decoder_optimizer.step()
           if idx%100 ==0:
               print("loss is", loss.item()/keyword.shape[0])
           avg_loss_per_iter += loss.item()/keyword.shape[0]
        print("loss for iter {} is {}".format(itr, avg_loss_per_iter/len(batch_data)))
        if (avg_loss_per_iter/len(batch_data))<best_loss:
            best_loss = avg_loss_per_iter/len(batch_data)
            torch.save(keyword_encoder.state_dict(), './data/keyword.mdl')
            torch.save(input_encoder.state_dict(), './data/input_encoder.mdl')
            torch.save(output_decoder.state_dict(), './data/output_decoder.mdl')
            with open("./data/log.txt", mode='w') as out:
                out.write("loss for iter {} is {}".format(itr, avg_loss_per_iter/len(batch_data)))



def prepare_batch_training_data(data=[], batch_size = 1):
    result = []
    key_data = []
    key_origin_len =[]
    input_sentences = []
    output_sentences = []
    for idx in range(0,len(data),4):
        temp_input_sentences = []
        for ith in range(4):
            # print(temp_input_sentences)
            key, sentence = data[idx+ith]
            key_data.append(pad(get_sentence_ints(key),3))
            temp = [False]*3
            temp[len(key)-1] = True
            key_origin_len.append(temp)
            input_sentences.append(pad(temp_input_sentences, 24))
            output_sentences.append(pad(get_sentence_ints(sentence),7))
            temp_input_sentences += get_sentence_ints(sentence) + [0]
    for i in range(0, len(data), batch_size):
        result.append((key_data[i : min(i+batch_size, len(data))], input_sentences[i: min(i+batch_size, len(data))], output_sentences[i: min(i+batch_size, len(data))], key_origin_len[i : min(i+batch_size, len(data))]))
    return  result

def get_sentence_ints(sentence):
    ch2int = word_embedder.ch2int
    result = list(map(lambda ch: ch2int[ch], sentence))
    return result

def pad(input, max_len):
    ch2int = word_embedder.ch2int
    return input + [ch2int['<PAD>']]*(max_len-len(input))


def read_training():
    data = []
    with codecs.open('./data/training.txt', 'r', 'utf-8') as fin:
        for line in fin:
            l = line.rstrip().split('\t')
            data.append((l[1], l[0]))
    return data


def predict(keywords):
    keyword_encoder = EncoderRNN(torch.from_numpy(word_embedder.get_word_embedding(512)), 512)
    input_encoder = EncoderRNN(torch.from_numpy(word_embedder.get_word_embedding(512)), 512)
    output_decoder = AttentionDecoderRNN(512)
    # keyword_encoder.load_state_dict(torch.load('./data/model/keyword.mdl', map_location=torch.device('cpu')))
    # input_encoder.load_state_dict(torch.load('./data/model/input_encoder.mdl', map_location=torch.device('cpu')))
    # output_decoder.load_state_dict(torch.load('./data/model/output_decoder.mdl',map_location=torch.device('cpu')))
    char_set = set()
    input_sentece = []
    for keyword in keywords:
        # keyword.shape = (1, len(keyword))
        keyword = torch.tensor( pad(get_sentence_ints(keyword),3), dtype=torch.long).unsqueeze(0)
        keyword_decoder_hidden = torch.zeros(2, 1, 512, dtype=torch.float).float()
        keyword_decoder_cell = torch.zeros(2, 1, 512, dtype=torch.float).float()
        # print(keyword.shape)
        keyword_decoder_outputs, (keyword_decoder_hidden, keyword_decoder_cell) = keyword_encoder(keyword.transpose(0,1), keyword_decoder_hidden, keyword_decoder_cell)
        #(keyword_decoder_outputs.shape = (len(keyword), 1, 1024))
        h0 = torch.cat((keyword_decoder_outputs[0, :, 0:512], keyword_decoder_outputs[0, :, 512:]),dim=1)

        # print(input_sentece)
        pad_input_sentence = torch.tensor(pad(input_sentece, 24)).long()

        input_decoder_hidden = torch.zeros(2, 1, 512, dtype=torch.float).float()
        input_decoder_cell = torch.zeros(2, 1, 512, dtype=torch.float).float()
        # (seq_len, 1, 1024)
        input_encoder_outputs, (input_decoder_hidden, input_decoder_cell) = input_encoder(pad_input_sentence.unsqueeze(1), input_decoder_hidden, input_decoder_cell)

        # (seq_len+1, 1, 1024)
        encoder_outputs = torch.cat((h0.unsqueeze(0), input_encoder_outputs), dim=0)

        # shape(encoder_mask) -> ( batch, seq_len+1,)  , here mask[i]=1 means i is not used
        encoder_mask = torch.cat((torch.zeros(1, 1, dtype=torch.long), pad_input_sentence.unsqueeze(0)),dim=1) == 5291

        hidden = torch.zeros(1, 1, 512, dtype=torch.float).float()
        cell = torch.zeros(1, 1, 512, dtype=torch.float).float()
        decoder_output = torch.zeros(1, 1, 512*3, dtype=torch.float).float()
        # print(encoder_outputs[:,0,:])
        for i in range(7):
            decoder_output, (hidden, cell) = output_decoder(decoder_output, hidden, cell, encoder_outputs, encoder_mask.transpose(0, 1))
            index = torch.topk(decoder_output[0],3)

            input_sentece.append(index[1][0])
            print(word_embedder.int2ch[index[1][0]], end = '')
            print('(',word_embedder.int2ch[index[1][1]] ,')','(',word_embedder.int2ch[index[1][2]] ,')', end='')
        input_sentece.append(0)
        print('\n')


if __name__ == '__main__':
    # predict(['春', '夏','秋','冬'])
    if torch.cuda.is_available():
        with torch.cuda.device(0):
            train()
    else:
        train()
