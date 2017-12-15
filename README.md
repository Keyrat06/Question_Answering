# Question Answering

 In Natural Language Processing, Question Answering is an important type of Information Retrieval task. Online question answering forums such as those managed by StackExchange allow users to post a question on a subject with the community responding with suitable answers. In the last few years, there has been a in their popularity and thus a corresponding explosion in the number of their users. The absence of an effective automated ability to refer to and reuse answers already available for previous posted questions, means that the community has to repeatedly spend time and energy in answering the same question. In this paper, we first explore a method for finding a related question to a posed question, given supervised data from the AskUbuntu forum. We then explore methods to try and transfer the learned model over to the AskAndroid forum where we do not have supervised data. 


### Question Retrieval
1. [CNN Model](/ourImplementation/CNN_model.py)
    *   This file contains the interfacing code for our Question Retreival CNN_Model
2. [util](/util/util.py)
    *   Probably the most important file in this repository! It contains the engine code for all models as well as the training code and all helper functions.
3. [LSTM Model](/ourImplementation/LSTM_model.py)
    *   This file contains the interfacing code for our Question Retreival CNN_Model

### Domain Adaptation
1. [bm25 & tf-idf](/ourImplementation/bm25_td-idf.py)
    *   This file contains the interface code for using our inhouse bm25 and tf-idf code (all implementation is in  util)
2. [Direct Transfer](/ourImplementation/direct_domain_adaptation.py)
    *   This file contains the interfacing code for our Direct Transfer for both LSTM and CNN models
3. [Advisarial Domain Adaptation](/ourImplementation/domain_adaptation_advisarial.py)
    *   This file contains the interfacing code for our Advisarial Domain Adaptation code
    
Advisarial Domain Adaptation was highly influenced by [Unsupervised Domain Adaptation by Backpropagation](https://arxiv.org/pdf/1409.7495.pdf). The following figure shows the Advisarial model used and was taken from this paper:
![alt text](/Figures/Advisarial_Net.png)

    
#### Notes
1. To run this code you must export PYTHONPATH=<path to root of this directory>
2. qa includes the code gathered from "Denoising Bodies to Titles: Retrieving Similar Questions with Recurrent Convolutional Models" and was highly influential to our code in this repository
3. data includes data gathered from https://github.com/taolei87/askubuntu and https://github.com/jiangfeng1124/Android and also required the downloading of the standford glove embedding that were to large to put in a git repository

