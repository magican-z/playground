import sys
import tiktoken


def run(text):
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = encoding.encode(text)
    for t in tokens:
        print(encoding.decode_single_token_bytes(t))

    for c in text:
        unicode_ = ord(c)
        utf8_ = bytes(c, encoding='utf8')
        print('word:{}, unicode:{}, utf8:{}'.format(c, unicode_, utf8_))


if __name__=="__main__":
    text = sys.argv[1]
    run(text)
