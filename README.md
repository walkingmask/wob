# walkingmask/wob
[World of Bits](http://proceedings.mlr.press/v70/shi17a/shi17a.pdf)に基づいた実験用リポジトリ。

(World of Bitsを完全に再現したものではない。)

MiniWoBを対象とし、[openai/universe](https://github.com/openai/universe)の`wob.mini`環境を使用する。


## 各ディレクトリの説明
### [agent](./agent)
[openai/universe-starter-agent](https://github.com/openai/universe-starter-agent)をベースにしたwob用のA3C学習用プログラム、デモデータを用いたBehavioral Cloningを行うためのプログラム、Success Rate(SR)を計測するためのプログラムなど。

### [expt](./expt)
実験について。概要、手順、実際の実験で作成したデモデータや記録。

### [misc](./misc)
実験用サーバのプロビジョニング用のメモなど。

### [recorder](./recorder)
デモの録画や、それをPythonオブジェクトに変換するためのプログラムなど。


## Tips
### 圧縮と転送
demo、ckptやlog等はファイルサイズが大きい傾向がある。これらを転送する時、圧縮すると楽。

```
# macでtar.bz2を扱う
tar ycvf demo.tar.bz2 --exclude='*/.*' demo # compress
tar yxvf demo.tar.bz2 # decompress
# Ubuntu
tar -jcvf demo.tar.bz2 --exclude=".*" demo # compress
tar -jxvf demo.tar.bz2 # decompress
```

### play2gif
エージェントのplayを画像として保存、[convertコマンド](https://qiita.com/sowd/items/832594dd22d99aebc8a2)でgifに変換する。

```
python agent/py/play.py wob.mini.HighlightText-v0 -s 100 -i ~/Desktop/play
convert -delay 20 -loop 0 ~/Desktop/play/play*.png ~/Desktop/play.gif
```

### SRのプロット
[gnuplot](https://qiita.com/noanoa07/items/a20dccff0902947d3e0c)を使って、次のようにグラフを出力し、保存する。

```
set datafile separator ","
set yrange [0:110]
set grid
set ylabel '% Human Performance'
set style fill solid border lc rgb "black"
set boxwidth 0.25 relative
set xtics rotate by -90
plot \
    '~/Desktop/expt/sr.txt' using ($0*4+0):2         with boxes lw 1 lc rgb "red"  title "Random",\
    '~/Desktop/expt/sr.txt' using ($0*4+1):3:xtic(1) with boxes lw 1 lc rgb "dark-yellow" title "SL",\
    '~/Desktop/expt/sr.txt' using ($0*4+2):4         with boxes lw 1 lc rgb "dark-green"  title "SL+RL"
replot
```

### SSH
GCEでpreemptiveインスタンスを使うと、起動毎にIPアドレスが変わる。sshやrsyncする時は、

```
IPADDR=x.x.x.x
ssh user@$IPADDR -i ~/.ssh/rsa
rsync -e "ssh -i ~/.ssh/rsa" ~/Desktop/demo/ user@$IPADDR:~/demo
```

のように、IPADDRを変数に格納しておいて、履歴を使って実行すると楽。

### wobのDocker Imageの中へアクセス
wobのDocker Imageには`universe`関連のソースコードが色々含まれており、とても参考になる。アクセスするには、Pythonからuniverseでwobのenvを起動しておいて、`docker exec`を使ってコンテナに入る。

```
python wob/misc/starter.py
docker exec -it CONTAINER_ID /bin/bash
```


## Todo
- Englishization
