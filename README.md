# Arabic-Speech-Embedding-Model
A CNN-based triplet Siamese network has been developed to generate Arabic speech embeddings. The embedding model was trained on a dataset comprising 1 million samples, sourced from various podcasts, and featuring speech from 198 distinct Arabic speakers spanning different dialects.

The trained embedding model is stored in the file siamese_network_Arabic_speech_embedding.h5. The python file si.py loads the model and uses the embeddings to train ML classifiers & a DNN for speaker identification. Users should adjust the references to the speaker.csv and si_mel_spec.csv files within the script to correspond with the dataset under consideration.

Kindly cite the paper titled "Unsupervised Arabic Speech Embedding Model for Speaker Identification" when using the emdedding model.
https://ieeexplore-ieee-org.aus.idm.oclc.org/document/10191576
