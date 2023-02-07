# Arabic-Speech-Embedding-Model
A CNN-based triplet Siamese network is built and used to generate Arabic speech embeddings. The embedding model was trained on 1 million samples, 
collected from different podcasts, of 198 Arabic speakers speaking different dialects. 

siamese_network_Arabic_speech_embedding.h5 --> Embedding model

si.py --> Python file, which loads the model and uses the embeddings to train ML classifiers & a DNN for speaker identification. The speaker.csv and 
si_mel_spec.csv files in the script should be changed based on the dataset used.
