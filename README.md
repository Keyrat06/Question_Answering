# Question Answering

### Question Retrieval
1. [CNN Model](../ourImplementation/CNN_model.py)
    *   This file contains the interfacing code for our Question Retreival CNN_Model
2. [util](../util/util.py)
    *   Probably the most important file in this repository! It contains the engine code for all models as well as the training code and all helper functions.
3. [LSTM Model](../ourImplementation/LSTM_model.py)
    *   This file contains the interfacing code for our Question Retreival CNN_Model

### Domain Adaptation
1. [bm25 & tf-idf](../ourImplementation/bm25_td-idf.py)
    *   This file contains the interface code for using our inhouse bm25 and tf-idf code (all implementation is in  util)
2. [Direct Transfer](../ourImplementation/direct_domain_adaptation.py)
    *   This file contains the interfacing code for our Direct Transfer for both LSTM and CNN models
3. [Advisarial Domain Adaptation](../ourImplementation/domain_adaptation_advisarial.py)
    *   This file contains the interfacing code for our Advisarial Domain Adaptation code
    
#### Notes
1. To run this code you must export PYTHONPATH=<path to root of this directory>
2. qa includes the code gathered from "Denoising Bodies to Titles: Retrieving Similar Questions with Recurrent Convolutional Models" and was highly influential to our code in this repository
3. data includes data gathered from https://github.com/taolei87/askubuntu and https://github.com/jiangfeng1124/Android and also required the downloading of the standford glove embedding that were to large to put in a git repository

