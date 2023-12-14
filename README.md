# CycleGANで自作のデータセットから画像生成
### データセットの作成
Google Colabで生成
https://drive.google.com/drive/folders/1td5iyoDN1BO57wOe76cLi2WVRZ7w8_hb?usp=sharing
今回は男の顔を女に変えるための画像を収集している

### モデル構築
10時間くらいかかったからColabでやった方がいいかも...
```
conda install tensorflow
pip install -q tfds-nightly matplotlib

git clone https://github.com/tensorflow/examples.git
```
実行
（入ってないライブラリ等あればインストールする必要があります）
```
python create_gan_model.py
```
### 画像生成

