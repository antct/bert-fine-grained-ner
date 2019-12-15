# bert-entity-typing

Fine-grained entity typing using BERT.

## network structure

(BERT + CharCNN) - (Mention Rep + Context Rep) - Fine-grained Types

## Production Env

`api.py` provide a simple flask api for model inference using cpu.

`stream.py` provide a multi-gpu way for inference with the help of [ShannonAI/service-streamer](https://github.com/ShannonAI/service-streamer).

## Dir Tree

```
.
├── api.py --- [simple flask api]
├── attention.py --- [attention module, dot-scaled & multi-head attention]
├── bert --- [bert base or bert large]
├── bert_base --- [bert pretrained model]
├── bert_large --- [bert pretrained model]
├── bilstm.py --- [rnn module]
├── cnn.py --- [cnn module, for char embedding]
├── config.py --- [config module]
├── ernie --- [ernie base]
├── ernie_base --- [ernie pretrained model]
├── fusion.py --- [fusion module]
├── log.py --- [logging module]
├── main.py --- [main entry, for training & fine_tuning & testing]
├── metric.py --- [metrice module, marco & micro P R F1]
├── model.py --- [model module, using bert directly or with BiLSTM+CRF]
├── network.png --- [network structure screenshot]
├── readme.md --- [readme]
├── stream.py --- [multi-gpu flask api]
├── data --- [data dir]
└── util.py --- [dataset module]
```
