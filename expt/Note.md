# Experiments
いくつかの実験についての記録。


## SR of wob
実験対象の5つのタスクのSR。

![./figs/wob_sr.pdf](./figs/wob_sr.pdf)


## expt1
ClickCollapsibleはいい感じだが、再現には至らなかった。

- Global Average Pooling実装
- bcはepoch=100
- SRは`universe-*.log`を使って計測
    - これは間違った行為だった
    - なぜならrewardなどの重要な情報の抜け落ち・重複が多かった

### SR
```
# Task,Random,SL,SL+RL
ClickCollapsible,27,11,97
ClickDialog2,25,30,23
GuessNumber,0,0,0
HighlightText,7,2,4
ResizeTextarea,2,2,1
```

![./figs/expt1.pdf](./figs/expt1.pdf)


### Diagnostics
各種diagnosticsについてメモ。

#### BC
- `loss`: 非常に振動する。htだけ見事に下降するが、0には至らない
- `step/sec`: 上昇して行く。安定してなくていいのだろうか？

#### A3C
- `fps`: バラツキはあるが、10~11fps前後。200kstepまで徐々に上がっていくケース、逆に80kstepがピークなケース、300kstepまでに山が二つあるケースと安定していなかった
- `vnc_updates_bytes`: 0~2e+4まで全く安定しない。0に張り付いているケースや、剣山のようになっているケースもあった
- `vnc_updates_n`: タスクによって上限下限は異なるが、それぞれ分散は`vnc_updates_bytes`ほどではなかった。こ`vnc_updates_bytes`が0だとこっちも0
- `vnc_updates_pixels`: `vnc_updates_bytes`と似た傾向
- `global_step/sec`: 主に平均128で振動していた
- `entropy`: 上昇する。緩やかに、急激に、範囲はタスクによる。通信がうまく行ってないときは0に張り付く
- `global_norm`: 下降傾向。うまくいっていないタスクは0に収束、ccでは平均20で大きく分散していた
- `policy_loss`, `value_loss`: cc以外はそれぞれ上昇、下降して収束に向かっているように見える。ただ、学習後半で不安定になるケースが多く見られた。学習のうまくいったccだけ振動していたのは謎
- `global_norm_var`: 上昇。崖のような上昇が多く見られた


## expt2
SRを`universe-*.log`ではなく、プログラムに直接集計させて出力するように実装し直したもの。

結局、expt_1と結果は変わらなかった。

結果が変わらなかったことから、この結果が今回の実験の最終結果として安定している、と言えるようになったではある。

### SR
```
# Task,Random,SL,SL+RL
ClickCollapsible,27,11,95
ClickDialog2,26,29,21
GuessNumber,0,0,0
HighlightText,7,1,4
ResizeTextarea,3,1,1
```

![./figs/expt2.pdf](./figs/expt2.pdf)


## expt3
expt2とほぼ同じだが、ソースコードを整理しながら実行したもの。expt1や2とだいぶ異なる結果が見れたが...。

### SR
```
# Task,Random,SL,SL+RL
ClickCollapsible,28,96,94
ClickDialog2,27,29,28
GuessNumber,0,7,10
HighlightText,7,0,16
ResizeTextarea,3,2,2
```

![./figs/expt3.pdf](./figs/expt3.pdf)


### Diagnostics
#### BC
- 所要時間: cc:5h, cd:2h50m, gn:3h20m, ht:5h30m, rt:4h
- validation lossを加えた。training lossと見比べると、かなり早い段階でof?しているような感じ(lossが上がっていく、急激に上がる)

#### A3C
- 所要時間: 2h/taskくらい
- GCE落ちる...これならGCE使わないべきだが、原因はなんだ？

#### Play
- 所要時間: 2h30m前後/task


## expt4
BCでの学習効果を確かめるために、BCなしA3Cのみ、他は同じ条件で実験した。現状、検証可能なタスクがClickCollapsibleだけなので、それだけを使用した。

```
a3c,ClickCollapsible,0.959386
a3c,ClickCollapsible,0.972879
a3c,ClickCollapsible,0.000000
```

結果は、最後の1つは死んでてダメだったが、上二つを見るとわかる通り、ClickCollapsibleではBCの効果は無さそうだった。より複雑なタスクでは違う結果が見られるかもしれない。


## expt5
bc時にデモデータの画像を正規化(0~255を0~1.0に変換すること)していなかった致命的な実装ミスがあったので、それを修正して再度1から実験した。結果はexpt3より微妙なことに。ただ、A3Cのスコアが下がってBCのスコアが上がった...気がする。BCのvalidation loss見ると、ほとんどのタスクで学習のほぼ初期で右肩上がりになっていたので、やっぱりover fittingしてるのかも。

### SR
```
# Task,Random,SL,SL+RL
ClickCollapsible,30,96,91
ClickDialog2,28,29,21
GuessNumber,0,8,2
HighlightText,7,5,5
ResizeTextarea,3,2,2
```

![./figs/expt5.pdf](./figs/expt5.pdf)

### Diagnostics
#### BC
- validation lossがほとんどのタスクで学習のほぼ初期で右肩上がり

#### A3C
- gnのvnc_updatesがいまいち不安定な感じなのはタスクの性質(画面変化が少ないから)なのだろうか
- ht,rtも同じような感じ、難易度の高いタスクほどvnc_*が悪い傾向があるような
- fpsが徐々に下がっていくなど


## expt6
bc時のエポック数を落とした実験。6epochsのckptでa3cした。SRを見るに、あまり変わらないか、悪くなっている。

### SR
```
# Task,SL,SL+RL
ClickCollapsible,96,86
ClickDialog2,30,27
GuessNumber,5,0
HighlightText,7,0
ResizeTextarea,1,1
```

### Diagnostics
特にいつもと変わりばえしないように感じる。

### メモ
この実験だけに言えることではないが、A3Cを使った実験では、ワーカー毎にtensorflowのログが作られるのでtensorboard用のsummaryが膨大になる。実験全てにおいてこれを保存してはおけないし、tensorboardを起動するのも遅く、解析に時間がかかる。表示も遅くてストレス。そしてtbをpdfに保存したりはできないという...。なんとかならんかな。


## expt7
ワーカー数を6に減らして、マシンの負荷を軽減して実験した。ckptにはexpt6のものを使用。3つのタスクに絞って、a3cも1回だけ学習。結果に大きな違いは見られなかった。Diagnostics見ると、vnc_updatesが地べたに張り付いたりはしていないので、その辺は良いのかなと。ただし、それでもhtのスコアが伸びていないあたり、問題はそこではない可能性が高い。

### SR
```
# Task,SL,SL+RL
ClickCollapsible,96,96
ClickDialog2,30,18
HighlightText,7,4
```


## expt8
poolingをGAPから普通のaverage pooling(2x2)に変更したもの。BCから再実行。bcのepochsは10。初めてhtのbcのスコアが改善された。先行研究に近い。しかし、a3cは全て0。原因は明確で、Diagnosticsを見るとvnc_updatesが0に張り付いているケースが多く、処理が重いことがわかる。GAPに比べてパラメータ数がかなり増えるのでそれのせいかなと。それにしても0になるものだろうか。

### SR
```
# Task,SL,SL+RL
ClickCollapsible,94,0
ClickDialog2,27,0
HighlightText,31,0
```
