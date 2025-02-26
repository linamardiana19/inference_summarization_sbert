# Extractive Summarization
This is the code that implements inference model SBERTSUM for SBERT + Inter-sentence Tranformer. The algorithm fine-tunes Google's language model IndoSBERT (Indonesia Sentence Bidirectional Encoder Representations from Transformers) for extractive text summarization, and enables the user to do end-to-end inference with the saved model. The pytorch model can be optimized for inference.

🌍 The original code for BERTSUM by Yang Liu could be found [here](https://github.com/nlpyang/BertSum)

**Python version**: the code is written in Python 3.11.4

**Package Requirements**: torch==2.5.1, sentence-transformers==2.2.2, nltk==3.9.1


## Training and Evaluation
This model is only for SBERTSUM at the inference level. For model training and model evaluation. 


## Choosing the Encoder
In order to run the model, you must download the checkpoint for SBERT + Classifier and IndoSBERT in the "models" repository. There are two option for the summarization layer. The first is the Simple Classifier, which is used in the original BertSum paper by Yang Liu. The second is the Deep Classifier, which uses deep feed-forward network as the summarization layer.


* SBERT + Deep Classifier : [link](......)

The saved checkpoints are optimized for inference. Compared to the fully loaded checkpoints produced during training phase, these saved checkpoints have 70% less file size, as well as show 20% faster inference speed.

## Running Inference
Type in the below code to run inference.

```
python predict_sbert.py 
```

This will prompt you to input the raw text that you wish to summarize. You could feed in the text that you wish to summarize, and press enter. The model will return to you the summary as text, with the number of sentences specified by the user. 
