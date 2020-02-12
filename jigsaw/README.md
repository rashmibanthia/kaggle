# Jigsaw Unintended Bias in Toxicity Classification 

This repo contains code for Kaggle competition - Jigsaw Unintended Bias in Toxicity Classification

My solution consisted of following models:
- Bert Large uncased (Trained on Google Colab TPUs) 
- Bert Base uncased  
- GPT2
- LSTM 

**bert_large.ipynb**
For Bert Large Uncased - I used Google colab TPUs (Tensorflow code), for the rest I used either Kaggle Kernels (P100) or AWS/GCP (V100 GPU) Pytorch/FastAI. 
For Bert Large, we need to setup the Google storage bucket and also connect to Google Drive. (Significant improvement in speed compared to GPUs.) This model scored around 0.938 - 0.939 on public LB. 
   
   ----
   
**rnn_lstm_5fold.ipynb**
This is Pytorch/FastAI, mostly based on kernel mentioned below. Uses Glove and FastText Embeddings. I added 5 fold processing and saving models for inference. It saves 20 models (Each fold has 4 epochs, and fit one cycle per epoch so 4 models per fold, we have 5 folds). It also uses the learning rate scheduler and weighted average for epochs. Public LB score 0.93795
    
----

**bert_fold0.ipynb**
Bert Base uncased - This is written in Pytorch, mostly based on kernel referenced below. There are few tricks that can be used to speed up the training time - Nvidia Apex (Mixed precision training), training similar length sequences in a batch. I have used either both/none/ or one of these in Bert base fold notebooks. Each fold had to be done separately due to GPU memory constraints and long training times. Inference results differ slightly when using Apex.  Public LB score ~0.941

----

**gpt2.ipynb**
This is also written in Pytorch, it uses OpenAI's GPT2 pretrained model - (See this discussion - https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/95220#latest-559430) The code for this is very similar to Bert base. They both use this repo https://github.com/huggingface/pytorch-pretrained-BERT.  I didn't have much time to tune this. It scored ~0.937 on public LB. 

----

**ensemble.ipynb**
This is used for inference. I had two ensembles for final submission. 

(1) 5 fold Bert Base +  another 5 fold Bert Base + 5 fold GPT 2 + LSTM 5fold + another LSTM 5 fold => My partner @YiTang helped me adjust weights for the 5 fold ensemble, since I had saved out of fold predictions. (Score public LB 0.94321, private LB 0.94263 )  I used faster inference technique, that allowed me to add multiple 5 fold models and reduced the inference time considerably. I used inference for similar length sequences in a batch (keeping track of ids) and then merging them back in order.  

(2) Bert Large (single) + Bert Base (single model) + 5 fold LSTM + GPT2 (single model) (Score public LB 0.94407, private LB 0.94248)   Pulbic LB Rank 105/3167  Final Rank 151.



    
My solution references these amazing kernels quite a bit - 
- https://www.kaggle.com/christofhenkel/how-to-preprocessing-for-glove-part2-usage
- https://www.kaggle.com/yuval6967/toxic-bert-plain-vanila
  
  
  
