# Federated Vision-Language Unlearning with Dual-view Knowledge Distillation
Architecture diagram：

Running：

1、requirement install：

```
pip install -r requirements.txt
```

2、Teacher model generation：

```
bash scripts/promptsrc/different_numshot_train.sh oxford_pets 1
```



3、run script：

```
CUDA_VISIBLE_DEVICES=0 scripts/federated-promptkd/fed_train_only.sh oxford_pets 1  
```

