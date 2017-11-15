# Question Answering

Follow instuctions in qa once you have exported PYTHONPATH and downloaded the dependencies and the data

run a command simmilar to python main.py --corpus ../../askubuntu-master/text_tokenized.txt --embeddings ../../askubuntu-master/vector/vectors_pruned.200.txt --train ../../askubuntu-master/train_random.txt --dev ../../askubuntu-master/dev.txt --test ../../askubuntu-master/test.txt --dropout 0.1 -d 400 --save_model first_model.pkl.gz
