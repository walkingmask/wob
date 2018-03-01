# Agent for wob
wob向けのA3Cエージェント。[openai/universe-starter-agent](https://github.com/openai/universe-starter-agent)に基づいてる。


## Usage

### Behavioral Cloning
`wob/record`で作成したデモデータセットを使って、Behavioral Cloningする。

```
$ cd wob/agent
$ chmod a+x ./bc.sh
$ nohup ./bc.sh /path/to/datasets /path/to/output &
```

データセットのパスと、ckpt等の保存先を指定。datasetsの中にある`wob.mini.*.bz2`ファイルの数だけ並行してBehavioral Cloningを実行する。

### A3C
Behavioral Cloningで学習したパラメータを使って、A3Cで学習する。

```
$ chmod a+x ./train.sh
$ nohup ./train.sh /path/to/bc_ckpt /path/to/output &
```

BCのckptと、ckpt等の保存先を指定。bcのckptの数x3回(wobの論文に基づいて)ループで走って、1回のループにつき複数ワーカがそのckptを使って走る。

### Run the learned agent
A3CまたはBCしたパラメータを使って、Agentにwobを実行させる。

```
$ chmod ./play.sh
$ nohup ./play.sh bc /path/to/bc_ckpt /path/to/output # bc
$ # or
$ nohup ./play.sh a3c /path/to/a3c_ckpt /path/to/output # a3c
$ # or
$ nohup ./play.sh rand /path/to/ckpt /path/to/output # rand
```

ckptのパスと、ログの保存先を指定。ckptの数だけ`play.py`が走る。

bcとa3cで実行モードが異なるのは、a3cではタスク1つにつきckptのディレクトリが3つある想定だから。

randモードでは、/path/to/ckptにあるタスク名のディレクトリ名だけを利用し、ckptは利用しない。

実行が終わると、`/path/to/output`に`universe-*.log`が保存される。このログに、SR(Success Rate)が記録されている。playに関するログを見たい場合は

```
$ # playに関する全てのログ
$ cat /path/to/output/universe-*.log | grep '\[wob player\]'
$ # タスク
$ cat /path/to/output/universe-*.log | grep '\[env\]'
$ # 使用したckpt
$ cat /path/to/output/universe-*.log | grep '\[ckpt\]'
$ # 記録されている全てのreward
$ cat /path/to/output/universe-*.log | grep '\[reward\]'
$ # SR
$ cat /path/to/output/universe-*.log | grep '\[SR\]'
```


## Tips
### モデルの定義と指定方法
エージェントの使用するネットワークモデルは、`wob/agent/py/models`に定義する。例えば、`TestModel`を`test_model.py`に定義したとする。これを使用するには、`models/__init__.py`に

```
from .test_model import TestModel
```

と記述し、`TestModel`を使用するモデルとして指定すれば良い。`bc.sh`であれば

```
./bc.sh /path/to/datasets /path/to/output -m TestModel
```

といった具合に。
