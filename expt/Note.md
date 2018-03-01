# Experiments
いくつかの実験についての記録。


## SR of wob
実験対象の5つのタスクのSR。

![](./figs/wob_sr.pdf)


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

![](./figs/expt1.pdf)


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

![](./figs/expt2.pdf)


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

![](./figs/expt3.pdf)
