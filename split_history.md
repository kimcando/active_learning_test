```
python imbalance_train.py --ratio "1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0" --split_order=0

python imbalance_train.py --ratio "1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0" --split_order='0_1'
```

### 0115
```nohup python imbalance_train.py --ratio "1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0" --split_order='balance1' > 1.txt
nohup python imbalance_train.py --ratio "0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5" --split_order='balance0.5' > 2.txt
nohup python imbalance_train.py --ratio "0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0" --split_order='imbal1' > 3.txt
nohup python imbalance_train.py --ratio "0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0" --split_order='imbal12' > 4.txt
nohup python imbalance_train.py --ratio "0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0" --split_order='imbal123' > 5.txt
nohup python imbalance_train.py --ratio "0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0" --split_order='imbal1234' > 6.txt
nohup python imbalance_train.py --ratio "0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0" --split_order='imbal12345' > 7.txt
nohup python imbalance_train.py --ratio "1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5" --split_order='imbal678910' > 8.txt
nohup python imbalance_train.py --ratio "0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0" --split_order='imbal123456' > 9.txt
nohup python imbalance_train.py --ratio "0.2, 0.8, 0.2, 0.8, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0" --split_order='imbal282851010101010' > 10.txt
```
* 1234 가 학습이 이상함.

### 0116
