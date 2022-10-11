#!/usr/bin/env python
# coding: utf-8

import re
from nltk.util import ngrams, pad_sequence, everygrams
from nltk.tokenize import word_tokenize
from nltk.lm import MLE, WittenBellInterpolated
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from csv import writer

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

data = [{'training' : 'data/real - aladin/ms-aladin-witness1-plagiarism.txt', 'testing' : 'data/real - aladin/ms-aladin-witness2-plagiarism.txt', 'output' : 'plagiarism-msaladin-w1-msaladin-w2.csv'},
            {'training' : 'data/real - aladin/ms-aladin-witness2-plagiarism.txt', 'testing' : 'data/real - aladin/ts-aladin-witness1-plagiarism.txt', 'output' : 'plagiarism-msaladin-w2-tsaladin-w1.csv'},
            {'training' : 'data/real - aladin/ts-aladin-witness1-plagiarism.txt', 'testing' : 'data/real - aladin/ts-aladin-witness2-plagiarism.txt', 'output' : 'plagiarism-tsaladin-w1-tsaladin-w2.csv'}]

models = ['MLE', 'Witten-Bell']

for item in data:
    print(item['training'], item['testing'])
    print()
    # Training data file
    train_data_file = item['training']
    # Testing data file
    test_data_file = item['testing']

    row_contents = ['token','plagiarism_score','context','ngram_length','language_model']
    append_list_as_row(item['output'], row_contents)


    # read training data
    with open(train_data_file) as f:
        train_text = f.read().lower()

    # apply preprocessing (remove text inside square and curly brackets and rem punc)
    train_text = re.sub(r"\[.*\]|\{.*\}", "", train_text)
    train_text = re.sub(r'[^\w\s]', "", train_text)
    print(len(train_text))
    # train_text = re.sub(' +', ' ', train_text)
    train_text = " ".join(train_text.split())
    print(len(train_text))

    for model in models:
        print(model)
        print()
        for n in range(2, 11):
            print(n)
            # pad the text and tokenize
            training_data = list(pad_sequence(word_tokenize(train_text), n, 
                                            pad_left=True, 
                                            left_pad_symbol="<s>"))

            # generate ngrams
            ngrams = list(everygrams(training_data, max_len=n))
            # print("Number of ngrams:", len(ngrams))

            # build ngram language models # 
            model_ = None
            if model == 'MLE':
                model_ = MLE(n)
            else:
                model_ = WittenBellInterpolated(n)

            model_.fit([ngrams], vocabulary_text=training_data)
            # print(model_.vocab)

            # Read testing data
            with open(test_data_file) as f:
                test_text = f.read().lower()
            
            test_text = re.sub(r"\[.*\]|\{.*\}", "", test_text)
            test_text = re.sub(r'[^\w\s]', "", test_text)
            test_text = " ".join(test_text.split())

            # Tokenize and pad the text
            testing_data = list(pad_sequence(word_tokenize(test_text), n, 
                                            pad_left=True,
                                            left_pad_symbol="<s>"))
            # print("Length of test data:", len(testing_data))

            # assign scores
            scores = []
            
            for i, item_ in enumerate(testing_data[n-1:]):
                context_words = testing_data[i-n:i]
                context_words.extend(testing_data[i:i+n+1])
                # s = model_.score(item_, testing_data[i:i+n-1])
                s = model_.score(item_, context_words)
                context = ' '.join(context_words)
                row_contents = [item_, s, context, n, model]
                append_list_as_row(item['output'], row_contents)
                scores.append(s)

            # print("scores: ", scores)

            scores_np = np.array(scores)

            # set width and height
            width = 8
            height = np.ceil(len(testing_data)/width).astype("int32")
            # print("Width, Height:", width, ",", height)

            # copy scores to rectangular blank array
            a = np.zeros(width*height)
            a[:len(scores_np)] = scores_np
            diff = len(a) - len(scores_np)

            # apply gaussian smoothing for aesthetics
            a = gaussian_filter(a, sigma=1.0)

            # reshape to fit rectangle
            a = a.reshape(-1, width)

            # format labels
            labels = [" ".join(testing_data[i:i+width]) for i in range(n-1, len(testing_data), width)]
            labels_individual = [x.split() for x in labels]
            labels_individual[-1] += [""]*diff
            labels = [f"{x:60.60}" for x in labels]

            # create heatmap
            fig = go.Figure(data=go.Heatmap(
                            z=a, x0=0, dx=1,
                            y=labels, zmin=0, zmax=1,
                            customdata=labels_individual,
                            hovertemplate='%{customdata} <br><b>Score:%{z:.3f}<extra></extra>',
                            colorscale="burg"))

            fig.update_layout({"height":height*28, "width":1000, "font":{"family":"Courier New"}})
            fig['layout']['yaxis']['autorange'] = "reversed"
            fig.write_html('interactive/' + item['output'].replace('.csv', '') + '-ngram-' + str(n) + '-' + model + ".html")
            # fig.show()
