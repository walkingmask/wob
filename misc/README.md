# Misc
サーバ関連など。


## ファイル等の説明
### tool
Slackへリポートを送るためのスクリプト等。

### [provisioning_linux.txt](./provisioning_linux.txt)
Linux用のプロビジョンングメモ。

### [provisioning_max.txt](./provisioning_max.txt)
Mac用のプロビジョニングメモ。

### record.sh
デモデータ作成をChromeとQuicktime Playerで実現しようとしたスクリプト。遺物。

### starter.py
universeでwobの環境を走らせるだけのスクリプト。プロビジョニングでDocker imageを引っ張ってくるのに使うなど。これを実行した後に、

```
$ docker exec -it CONTAINER_ID /bin/bash
```

で、wobのDocker containerに入ることができる。

### wob_envs.txt
universeに登録されているwobのenv一覧。[`openai/universe/universe/__init__.py`](https://github.com/openai/universe/blob/master/universe/__init__.py#L1483)を見ると、他にもwobのenvがありそうだが、動かすことができなかった。


## GCEの利用についてのメモ

### 各種設定
|Key|Value|
|---|---|
|Machine type|n1-highcpu-32|
|Zone|us-east1-b|
|Storage|30G|
|Preemptive|On|

### Machine type
`n1-highcpu-32`を使用した。このインスタンスはメモリサイズを抑えてCPUコア数を多く必要とする場合に使用する。`32vCPU`、`28.8 GB Mem`。

### Zone
料金的な理由から`us-east1-b`を使用。

### Firewall
HTTPは使わないが、TensorboardやVNCのために設定する。[ここ](https://console.cloud.google.com/networking/firewalls)(アカウントやプロジェクトの切り替えを忘れずに)にアクセスして、ファイアーウォールルールを作成する。`allow-tb-vnc`など名前を適当につける。Networkはdefault。優先度は適当に`99*`とか。トラフィック方向は上り。アクションは許可。タグなどは必要に応じて設定。ソースフィルタは特にこだわりがなければ`0.0.0.0/0`。プロトコルとポートは、場合によるが`tcp:12345,tcp:5900`のように。作成すると、インスタンス側のネットワーク設定を特にいじってなければすぐに適用されるはず。

### Storage
GCEには[無料階層](https://cloud.google.com/free/?hl=ja)がある。ストレージが30GBまでは無料なので、それを使う。

### Preemptive
`n1-highcpu-32`インスタンスは、次のような料金になっている。

- 通常: `$1.1344/hour`
- preemptive: `$0.2400/hour`

preemptiveインスタンスは約8割引で使用可能。ただし、以下の制約がある。

- 最大起動時間は24時間
- 強制終了されることがある

これを考慮しても魅力的な値段なので、これに対応するために、学習中のパラメータやログなどを随時保存したり、終了される前に避難処理(shutdown-script)といったことをする。

#### shutdown-script
インスタンスの設定のカスタムメタデータで`shutdown-script:/path/to/script`のような感じで設定できる。scriptは実行可能形式で、強制終了される前30秒間で完了する必要がある。運用前にしっかりテストすることを推奨。テストはインスタンスを起こしたり落としたりするので面倒。

### SSH
インスタンス作成の時に公開鍵をコピペしておくと楽。普段使っているユーザー名で`ssh user@ipaddr -i ~/.ssh/rsa.pub`と簡単にアクセスできる。ただし、preemptiveインスタンスは毎回IPアドレスが変わることに注意。

### gcloudコマンド
GCEをブラウザで操作すると結構時間がかかったので、gcloudコマンドなどを使うと良いかもしれない。
