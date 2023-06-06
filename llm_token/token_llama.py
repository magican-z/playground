import sys
import os
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> list[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: list[int]) -> str:
        res = []
        for ti in t:
            ri = self.sp_model.decode([ti])
            print('{} -> {}'.format(ti, ''.join(ri)))
            res.extend(ri)
        return res
        #return self.sp_model.decode(t)

def run(text):
    tokenizer = Tokenizer(model_file_path)
    encoding = tokenizer.encode(text, bos=False, eos=False)
    print('encoding:{}'.format(encoding))
    tokenizer.decode(encoding)

if __name__=='__main__':
    run(sys.argv[1])
