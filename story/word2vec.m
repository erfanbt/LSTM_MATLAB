%% 
clear;
%% 
filename = "belling_the_cat.txt";
str = extractFileText(filename);
% textData = split(str,newline);
documents = tokenizedDocument(str);
%% 
emb = trainWordEmbedding(documents, ...
    'Dimension',50, ...
    'MinCount',1, ...
    'NumEpochs',10);
%% 
filename = "belling_the_cat.vec";
writeWordEmbedding(emb,filename);
%% 
words = emb.Vocabulary;
V = word2vec(emb,words);
XY = tsne(V);
textscatter(XY,words);
