import numpy as np
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm


def filtering(train_data, test_data):
    # view data
    # print(train_data[:5])
    # print(train_data['document'].nunique(), train_data['label'].nunique())
    print(print('총 샘플의 수 :', len(train_data), ',', len(test_data)))
    # print(train_data.groupby('label').size().reset_index(name='count'))

    # remove eng, space
    train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    test_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    # print(train_data[:5])

    # check unique
    train_data.drop_duplicates(subset=['document'], inplace=True)
    test_data.drop_duplicates(subset=['document'], inplace=True)

    # check null
    print(train_data.isnull().sum())
    print(test_data.isnull().sum())

    train_data = train_data.dropna(how='any')
    test_data = test_data.dropna(how='any')
    print('총 샘플의 수 :', len(train_data), ',', len(test_data))
    return train_data, test_data


def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if (len(sentence) <= max_len):
            count = count + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s' % (max_len, (count / len(nested_list)) * 100))


def token(train_data, test_data):
    # tokenizer
    stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
    okt = Okt()
    X_train = []
    for sentence in tqdm(train_data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
        X_train.append(stopwords_removed_sentence)

    X_test = []
    for sentence in tqdm(test_data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True)  # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
        X_test.append(stopwords_removed_sentence)

    #oov = out of vocabulary
    #이렇게 되면 validation / test dataset에 fit_on_texts 했을 때 OOV token에 대해 1이라는 숫자를 부여해준다.
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    threshold = 3
    total_cnt = len(tokenizer.word_index)  # 단어의 수
    rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    vocab_size = total_cnt - rare_cnt + 1
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)
    # print(tokenizer.word_index)

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print('단어 집합(vocabulary)의 크기 :', total_cnt)
    print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' % (threshold - 1, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

    #embeding
    vocab_size = total_cnt - rare_cnt + 1
    tokenizer = Tokenizer(vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    y_train = np.array(train_data['label'])
    y_test = np.array(test_data['label'])

    drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
    X_train = np.delete(X_train, drop_train, axis=0)
    y_train = np.delete(y_train, drop_train, axis=0)

    print('리뷰의 최대 길이 :', max(len(review) for review in X_train))
    print('리뷰의 평균 길이 :', sum(map(len, X_train)) / len(X_train))

    max_len = max(len(review) for review in X_train)
    print(below_threshold_len(max_len, X_train))

    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    return X_train, X_test, y_test, y_train, max_len, vocab_size, tokenizer
