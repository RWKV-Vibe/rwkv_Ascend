import os
import urllib
import tempfile
import urllib


class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name=None):
        vocab = file_name if file_name else self.get_vocab()
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(vocab, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)
    
    def get_vocab(self):
        VOCAB_NAME = "rwkv_vocab_v20230424.txt"
        VOCAB_SRC = [
            "https://www.modelscope.cn/models/EliwiiKeeya/RWKV-x060-World-1B6-v2.1-20240328-ctx4096/resolve/master/rwkv_vocab_v20230424.txt"
        ]
        temp_dir = tempfile.gettempdir()
        temp_vocab_path = os.path.join(temp_dir, "mindnlp", "rwkv6")
        temp_vocab = os.path.join(temp_vocab_path, VOCAB_NAME)

        if os.path.exists(temp_vocab) and os.path.getsize(temp_vocab) > 0:
            print("Use cached vocab: " + temp_vocab)
            return temp_vocab
        else:
            print("Download the vocab as: " + temp_vocab)
            if not os.path.exists(temp_vocab_path):
                os.makedirs(temp_vocab_path)

            for url in VOCAB_SRC:
                try:
                    urllib.request.urlretrieve(url, temp_vocab)
                except Exception as e:
                    print(e)
                else:
                    return temp_vocab
            else:
                raise(RuntimeError("Download failed."))

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()
