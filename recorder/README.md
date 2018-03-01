# Recording tools to create demonstrations for wob
wobタスクの操作を記録してデモデータにするためのツール。


## Usage
使い方。

### Recording wob demonstration
デモンストレーションを録画するには、`record.sh`を使う。前提として、Dockerが使える状態であること。使い方は

```
$ cd wob/recorder
$ source ./record.sh
$ record -t wob.mini.ClickCollapsible-v0 -s 100 # specify a task on command line
```

`-s 100`で、タスクを実行する時間を指定できる。`Screen Sharing`のパスワードは`openai`で、終了ダイアログが出たら`Screen Sharing`は手動で終了する必要がある。

### Convert demos to datasets
`record.sh`で記録したデモは、fbsファイル形式になっていて取り扱いにくいので、Pyhtonオブジェクト形式にしてpicklし、圧縮する。使い方は

```
$ source ./demos2datasets.sh
$ d2d -i /path/to/demos -o /path/to/datasets
```

### Utils
作成したデータの中を確認するためのプログラムが`py/utils.py`にある。使用する為にはopencvが必要。

```
$ # utils.pyを実行してREPLを開く
$ python -i py/utils.py ~/Desktop/demos/wob.mini.ClickCollapsible-v0/1518170452-pkggly1mtlyc7i-1/
>>> # demoに含まれるフレームとアクションの数を数える
>>> count(reader)
the total number of frames is 777
the total number of actions is 328
>>> # readerをリロードする
>>> reload()
>>> # demoのフレームと行動を1フレームずつ見る
>>> # qで終了、それ以外のキーでフレームを進める
>>> play_demo(reader)
>>> reload()
>>> # demoをmov(動画ファイル)に変換する
>>> demo2mov(reader, '~/Desktop/demo.mov')
>>> # datasetの中身を検証する
>>> play_dataset('~/Desktop/datasets/wob.mini.ClickCollapsible-v0.bz2')
>>> exit()
```
