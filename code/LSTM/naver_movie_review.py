import pickle
import urllib.request
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import dataFiltering as d


def openFile():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                               filename="code/LSTM/data/ratings_train.txt")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                               filename="code/LSTM/data/ratings_test.txt")


def main():
    # filtering
    train_data = pd.read_table('code/LSTM/data/ratings_train.txt')
    test_data = pd.read_table('code/LSTM/data/ratings_test.txt')
    train_data, test_data = d.filtering(train_data,test_data)

    # tokenizer
    X_train, X_test, y_test, y_train, max_len, vocab_size, tokenizer = d.token(train_data, test_data)

    # #model
    # embedding_dim = 100
    # hidden_units = 128
    # epoch = 20
    # batch = 64
    #
    # model = Sequential()
    # model.add(Embedding(vocab_size, embedding_dim))
    # model.add(LSTM(hidden_units))
    # model.add(Dense(1, activation='sigmoid'))
    #
    # # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    # mc = ModelCheckpoint('code/LSTM/data/best_model.h5', monitor='val_acc', mode='max', verbose=2, save_best_only=True)
    #
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    # history = model.fit(X_train, y_train, epochs=epoch, callbacks=[mc], batch_size=batch, validation_split=0.2)
    #
    # #model end

    # test
    # loaded_model = load_model('code/LSTM/data/best_model.h5')
    # print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

    with open('code/LSTM/data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle)


if __name__ == '__main__':
    openFile()
    main()


