Dependencies,
```
pytorch 1.0.0
pandas
tqdm
xlrd (pip install xlrd)
bert-pytorch (pip install pytorch-pretrained-bert)
```

#### **To train a model**

To train a LSTM model, run the following command, (4 classification need add --ntag 4)

```
python main.py --batch_size 1024 --config lstm --encoder 0 --mode 0
```

To train a CNN model, run the following command, (4 classification need add --ntag 4)
```
python main.py --batch_size 1024 --config cnn --encoder 1 --mode 0
```

To train a BERT model, run the following command, (4 classification need add --ntag 4)
```
python bert_classifier.py --batch_size 4 --max_epochs 10 --max_seq_length 500 --max_sent_length 70 --mode 0
```

To train a GCN based model, run the following command, (4 classification need add --ntag 4)
```
python main.py --batch_size 32 --max_epochs 10 --config gcn --max_sent_len 50 --encoder 2 --mode 0
```

To train a GAT based model, run the following command, (4 classification need add --ntag 4)

```
python main.py --batch_size 32 --max_epochs 10 --config gat --max_sent_len 50 --encoder 3 --mode 0
```

To train a GAT based model with 2 attention, run the following command, (4 classification need add --ntag 4)

```
python main.py --batch_size 32 --max_epochs 10 --config gat_attn2h --max_sent_len 50 --encoder 4 --mode 0
```

To train a GAT based model with 3 attention, run the following command, (4 classification need add --ntag 4)

```
python main.py --batch_size 32 --max_epochs 10 --config gat_attn3h --max_sent_len 50 --encoder 5 --mode 0
```

To train a DeepGCN model, run the following command, (4 classification need add --ntag 4)

```
python main.py --batch_size 32 --max_epochs 10 --config deepgcn --max_sent_len 50 --encoder 6 --mode 0
```

#### **To test the accuracy of the model on the out of domain test set, run the following command**,

For the LSTM model,  (4 classification need add --ntag 4)
```
python main.py --batch_size 1024 --encoder 0 --model_file model_lstm.t7 --mode 1
```

For the CNN model,  (4 classification need add --ntag 4)
```
python main.py --batch_size 1024 --encoder 1 --model_file model_cnn.t7 --mode 1
```

For the Bert model,  (4 classification need add --ntag 4)
```
python bert_classifier.py --batch_size 4 --model_file model_bert.t7 --max_seq_length 500 --max_sent_length 70 --mode 1
```

For the GCN model,  (4 classification need add --ntag 4)
```
python main.py --batch_size 32 --max_sent_len 50 --encoder 2 --model_file model_gcn.t7 --mode 1
```

For the GAT model, (4 classification need add --ntag 4)
```
python main.py --batch_size 32 --max_sent_len 50 --encoder 3 --model_file model_gat.t7 --mode 1
```

For the GAT model with 2 attention head, (4 classification need add --ntag 4)
```
python main.py --batch_size 32 --max_sent_len 50 --encoder 4 --model_file model_gat_attn2h.t7 --mode 1
```

For the GAT model with 3 attention head, (4 classification need add --ntag 4)

```
python main.py --batch_size 32 --max_sent_len 50 --encoder 5 --model_file model_gat_attn3h.t7 --mode 1
```

For the DeepGCN model , (4 classification need add --ntag 4)

```
python main.py --batch_size 32 --max_sent_len 50 --encoder 6 --model_file model_deepgcn.t7 --mode 1
```



### 
### 
