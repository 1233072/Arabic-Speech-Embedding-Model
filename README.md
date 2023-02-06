# Arabic-Speech-Embedding-Model
This model is trained on 1 million speech samples from Arabic speakers.
A CNN-based triplet Siamese network was built for Arabic speech embedding model which is used to extract speech characteristics (i.e., features). The 
embedding model was trained on 1 million samples of Arabic speakers speaking different dialects. The feature network was used for speaker identification.

siamese_network_Arabic_speech_embedding.h5 --> Embedding model
si.py --> Python file which loads the model and uses the embeddings to train ML classifiers & a DNN for speaker identification. The speaker.csv and 
si_mel_spec.csv files in the script should be changed based on the dataset you will use.
