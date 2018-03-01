# Network Models
# モデルを簡単に切り替えられるように、各モデルは別個のファイルで定義して、ここに宣言し
# import models; getattr(models, name) で呼び出すように実装している

from .v1 import FFPolicy
from .no_gap import FFPolicy2
from .v0 import FFPolicy0
