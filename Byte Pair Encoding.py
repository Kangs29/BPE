
# coding: utf-8

# In[3]:

import os
import re, collections

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r"\,", " \, ", string) 
    string = re.sub(r"\.", " \. ", string) 
    string = re.sub(r"\!", " \! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()


def get_vocab(filename):
    vocab = collections.defaultdict(int)
    with open(filename, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            words = clean_str(line).strip().split()
            for word in words:
                vocab[' '.join(list(word.lower())) + ' </w>'] += 1
    return vocab


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word) # pair 기준으로 단어를 다시 합쳐버린다.
        v_out[w_out] = v_in[word]
    return v_out


def get_tokens_from_vocab(vocab):
    tokens_frequencies = collections.defaultdict(int)
    vocab_tokenization = {}
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq
        vocab_tokenization[''.join(word_tokens)] = word_tokens
    return tokens_frequencies, vocab_tokenization

def measure_token_length(token):
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    else:
        return len(token)

def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
    
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token]

# 이 함수안에 넣는 것은 아니고 필요할 경우, if-else문으로 tokenized_word함수를 사용할 지, 안할지 결정
# -> 모델에 사용 시
#    if string in sorted_tokens:
#        return [string]  # 이거 안할수없나... 나중에 찾아보자
    
    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token.replace('.', '[.]')) # . -> [.], 
        
        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]

        substring_start_position = 0
        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position:substring_end_position]
            string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
        break
    return string_tokens


# In[4]:

os.chdir(r'C:\Users\조강\Desktop\training-parallel-nc-v9\training')
vocab = get_vocab('news-commentary-v9.de-en.en')
tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)

num_merges = 200
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs,key=pairs.get)
    vocab = merge_vocab(best, vocab)

    tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)

    if i % 1000 == 0:
        print('Iter: {}'.format(i))

        
sorted_tokens_tuple = sorted(tokens_frequencies.items(), 
                             key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]


# In[9]:

print(' - Testing Byte-Pair Encoding')
s='\!</w>'
print('    word :',s,'- BPE :',tokenize_word(string=s, 
                          sorted_tokens=sorted_tokens, unknown_token='</u>'))
s='operation</w>'
print('    word :',s,'- BPE :',tokenize_word(string=s,
                                   sorted_tokens=sorted_tokens, unknown_token='</u>'))


# In[1]:

# Transformer byte pair encoding


# In[2]:

import os
import re, collections

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r"\,", " \, ", string) 
    string = re.sub(r"\.", " \. ", string) 
    string = re.sub(r"\!", " \! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()


def get_vocab(filenames):
    vocab = collections.defaultdict(int)
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as fhand:
            for line in fhand:
                words = clean_str(line).strip().split()
                for word in words:
                    vocab[' '.join(list(word.lower())) + ' </w>'] += 1 ['t h e</w>']
    return vocab


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word) # pair 기준으로 단어를 다시 합쳐버린다.
        v_out[w_out] = v_in[word]
    return v_out


def get_tokens_from_vocab(vocab):
    tokens_frequencies = collections.defaultdict(int)
    vocab_tokenization = {}
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq
        vocab_tokenization[''.join(word_tokens)] = word_tokens
    return tokens_frequencies, vocab_tokenization

def measure_token_length(token):
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    else:
        return len(token)

def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
    
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token]

# 이 함수안에 넣는 것은 아니고 필요할 경우, if-else문으로 tokenized_word함수를 사용할 지, 안할지 결정
# -> 모델에 사용 시
#    if string in sorted_tokens:
#        return [string]  # 이거 안할수없나... 나중에 찾아보자
    
    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token.replace('.', '[.]')) # . -> [.], 
        
        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]

        substring_start_position = 0
        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position:substring_end_position]
            string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
        break
    return string_tokens


# In[4]:

# German 37000 tokens
os.chdir(r'C:\Users\조강\Desktop\Transformer\train_parallel\English-German')
filenames = ["commoncrawl.de-en.de","europarl-v7.de-en.de","news-commentary-v9.de-en.de"]
vocab = get_vocab(filenames)
tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)

num_merges = 37000
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs,key=pairs.get)
    vocab = merge_vocab(best, vocab)

    tokens_frequencies, vocab_tokenization = get_tokens_from_vocab(vocab)

    if i % 1000 == 0:
        print('Iter: {}'.format(i))

        
#sorted_tokens_tuple = sorted(tokens_frequencies.items(), 
#                             key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
#sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]


# In[42]:

os.chdir(r"C:/Users/조강/Desktop")
with open("Tranformer_BPE_Deutsch.txt",'w',encoding='utf-8') as f:
    for lines in tokens_frequencies:
        f.write(lines+'\t'+str(tokens_frequencies[lines])+'\n')


# In[43]:

os.chdir(r"C:/Users/조강/Desktop")
with open("Tranformer_Vocab_Deutsch.txt",'w',encoding='utf-8') as f:
    for lines in vocab_tokenization:
        f.write(lines+'\t'+",".join(vocab_tokenization[lines])+'\n')


# In[38]:

sorted_tokens_tuple = sorted(tokens_frequencies.items(), 
                             key=lambda item: (measure_token_length(item[0]), item[1]), reverse=True)
sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]


# In[39]:

sorted_tokens


# In[47]:

# English 37000 tokens
os.chdir(r'C:\Users\조강\Desktop\Transformer\train_parallel\English-German')
filenames = ["commoncrawl.de-en.en","europarl-v7.de-en.en","news-commentary-v9.de-en.en"]
vocab_ = get_vocab(filenames)

min_count=5

vocab_en={}
for word in vocab_:
    if vocab_[word]>=min_count:
        vocab_en[word]=vocab_[word]
        
print("vocab :",len(vocab_en))

tokens_frequencies_en, vocab_tokenization_en = get_tokens_from_vocab(vocab_en)



num_merges = 37000
for i in range(num_merges):
    pairs = get_stats(vocab_en)
    if not pairs:
        break
    best = max(pairs,key=pairs.get)
    vocab_en = merge_vocab(best, vocab_en)

    tokens_frequencies_en, vocab_tokenization_en = get_tokens_from_vocab(vocab_en)

    if i % 1000 == 0:
        print('Iter: {}'.format(i))


# In[48]:

os.chdir(r"C:/Users/조강/Desktop")
with open("Tranformer_BPE_English.txt",'w',encoding='utf-8') as f:
    for lines in tokens_frequencies_en:
        f.write(lines+'\t'+str(tokens_frequencies_en[lines])+'\n')


# In[49]:

os.chdir(r"C:/Users/조강/Desktop")
with open("Tranformer_Vocab_English.txt",'w',encoding='utf-8') as f:
    for lines in vocab_tokenization_en:
        f.write(lines+'\t'+",".join(vocab_tokenization_en[lines])+'\n')


# In[51]:

def tokenize_word(string, sorted_tokens, unknown_token='</u>'):
    
    if string == '':
        return []
    if sorted_tokens == []:
        return [unknown_token]

# 이 함수안에 넣는 것은 아니고 필요할 경우, if-else문으로 tokenized_word함수를 사용할 지, 안할지 결정
# -> 모델에 사용 시
#    if string in sorted_tokens:
#        return [string]  # 이거 안할수없나... 나중에 찾아보자
    
    string_tokens = []
    for i in range(len(sorted_tokens)):
        token = sorted_tokens[i]
        token_reg = re.escape(token.replace('.', '[.]')) # . -> [.], 
        
        matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
        if len(matched_positions) == 0:
            continue
        substring_end_positions = [matched_position[0] for matched_position in matched_positions]

        substring_start_position = 0
        for substring_end_position in substring_end_positions:
            substring = string[substring_start_position:substring_end_position]
            string_tokens += tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            string_tokens += [token]
            substring_start_position = substring_end_position + len(token)
        remaining_substring = string[substring_start_position:]
        string_tokens += tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
        break
    return string_tokens

