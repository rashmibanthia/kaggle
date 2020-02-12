# Google Quest Solution (wip)

This is the code for my solution (rank 44/1571)  to the Google Quest Q&A Labeling challenge on Kaggle. This NLP competition requires us to predict the scores given by human raters to questions and answers on various Stack Exchange Q&A websites. The questions and answers are scored on 30 dimensions, e.g. question_well_written, answer_well_written, answer_helpfu,l	answer_level_of_information, answer_plausible, 'question_multi_intent', 'question_not_really_a_question' etc. 

#### bert_use_crawl.ipynb 
(All code in PyTorch) 
I had two separate models for question and answers. This file has training code for both these models. 
This is a fine tuned bert model - it includes the averaging of last 4 hidden layers + CLS token. 
Apart from that my model also includes, 
 - USE features 
 - crawl embeddings
