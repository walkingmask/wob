# Experiments
実際の実験で作成したデモデータや、手順、再現実験について。


## 用語の定義
- デモンストレーション: 人間のMiniWoBの実行記録
- デモデータ: デモンストレーションをフレーム-行動ペアに変換したもの


## 実験概要
5つのMiniWoBタスクについて追実験する。デモデータを作成し、Behavioral Cloning、A3Cによる学習を行う。学習後、SRを計測する。


## ファイルの説明
### [Note.md](./Note.md)
いくつかの実験についての記録。

### demo.tar.bz2
記録したデモンストレーション(.fbsなど)を圧縮したもの。

### datasets/wob.mini.*
作成したデモデータを圧縮したもの。


## 実験環境
- デモデータの作成
    - OSX 10.13.2
    - miniconda3-4.3.27
- BC, A3C, Play
    - n1-highcpu-32, 学科VM(24vCPU,24GMEM)
    - Ubuntu 16.04
    - anaconda3-4.1.1
- Docker 17.12.0-ce


## 実験方法
次のような手順で行う。

1. デモデータの作成
1. Behavioral Cloning
1. A3Cでの訓練
1. SRの計測

具体的には

1. `wob/recorder/record.sh`で、デモの録画
1. `wob/recorder/demos2datasets.sh`で、デモをPythonオブジェクトに変換して圧縮
1. サーバにでもデータを転送
1. `wob/agent/bc.sh`で、デモデータを使ってBehavioral Cloning
1. `wob/agent/train.sh`で、A3Cを使った強化学習
1. `wob/agent/play.sh`で、SRを計測


## 再現実験
`datasets/wob.mini.*`を使ってサーバ上で再現実験する流れ。

デモンストレーションのSRは以下の通り。これは、作成時の`universe-*.log`から集計した(よくない方法だった)。不運にもログは削除してしまった。

```
demo,ClickCollapsible,0.9954545454545455
demo,ClickDialog2,0.9972067039106145
demo,GuessNumber,0.9805825242718447
demo,HighlightText,0.9856459330143541
demo,ResizeTextarea,0.9846153846153847
```

以下は、実際に実験する流れ。`play.sh`は学科VMとGCEの両方を利用して実行した場合。

```
# Behavioral Cloning
cd wob/agent
chmod a+x ./bc.sh
nohup ./bc.sh ~/wob/expt/datasets ~/bc -e 2 &

# collect bc ckpt
cd && mkdir bc_ckpt
for task in `ls bc`; do
    [ "$task" = "log" ] && continue
    mkdir -p bc_ckpt/$task
    ckpt=`ls bc/$task/ckpt | grep 100`
    for ckpt_ in $ckpt; do
        cp bc/$task/ckpt/$ckpt_ bc_ckpt/$task/
    done
done

# A3C
cd wob/agent
chmod a+x ./train.sh
nohup ./train.sh ~/bc_ckpt ~/a3c &

# collect a3c ckpt
cd && mkdir a3c_ckpt
for task in `ls a3c`; do
    for i in `seq 3`; do
        mkdir -p a3c_ckpt/$task/$i
        cp a3c/$task/$i/train/model.ckpt* a3c_ckpt/$task/$i/
    done
done

# Play
cd wob/agent
chmod a+x ./play.sh

# random
nohup ./play.sh rand ~/a3c_ckpt ~/play/rand &

# bc
nohup ./play.sh bc ~/bc_ckpt ~/play/bc &

# a3c
# split ckpt
for task in `ls a3c_ckpt`; do
    mkdir -p a3c_ckpt_2/$task
    mv a3c_ckpt/$task/3 a3c_ckpt_2/$task/
done
# in vm
nohup ./play.sh a3c ~/a3c_ckpt ~/play/a3c &
# in gce
nohup ./play.sh a3c ~/a3c_ckpt_2 ~/play/a3c &

# collect all data into local:~/Desktop/play for analysis

# count SR
for mode in `ls ~/Desktop/play/`; do
    for log in `ls ~/Desktop/play/$mode`; do
        task=`cat ~/Desktop/play/$mode/$log | grep "\[env\]" | cut -d' ' -f6` 
        task_=${task##*.}
        sr=`cat ~/Desktop/play/$mode/$log | grep "\[SR\]" | cut -d' ' -f18` 
        echo "$mode,${task_:0:-3},$sr" >>~/Desktop/sr.txt
    done
done
# if can not count above, use below
log=/path/to/universe-xxxx.log
num_rewards=`cat $log | grep '\[reward\] | wc -l | tr -d ' '
num_positive_rewards=`cat $log | grep '\[reward\]' | grep -v ' -1.0' | wc -l | tr -d ' '`
echo $((1.0*num_positive_rewards/num_rewards))

# normalize
cd ~/Desktop
python <<EOL >>./sr_normalized.txt
with open('./sr.txt', 'r') as f:
    data = f.readlines()

SR = dict()

for d in data:
    mode, task, sr = d.strip().split(',')
    sr = float(sr)
    if not task in SR:
        SR[task] = dict()
    if mode in SR[task]:
        SR[task][mode] = max(SR[task][mode], sr)
    else:
        SR[task][mode] = sr

print('# Task,Random,SL,SL+RL')
for task in SR.keys():
    sr = SR[task]
    demo = SR[task]['demo']
    def normalize(x): return int((x/demo)*100)
    print("%s,%d,%d,%d"
        % (task, normalize(sr['rand']), normalize(sr['bc']), normalize(sr['a3c'])))
EOL

# plot 
gnuplot -e '
set datafile separator ",";
set yrange [0:110];
set grid;
set ylabel "% Human Performance";
set style fill solid border lc rgb "black";
set boxwidth 0.25 relative;
set xtics rotate by -90;
plot
    "~/Desktop/sr_normalized.txt" using ($0*4+0):2         with boxes lw 1 lc rgb "red"  title "Random",
    "~/Desktop/sr_normalized.txt" using ($0*4+1):3:xtic(1) with boxes lw 1 lc rgb "dark-yellow" title "SL",
    "~/Desktop/sr_normalized.txt" using ($0*4+2):4         with boxes lw 1 lc rgb "dark-green"  title "SL+RL";
replot;
'
```
