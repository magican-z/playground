import sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "model_file_path",
    local_files_only=True,
    trust_remote_code=True)

tokens = tokenizer.tokenize(sys.argv[1])
print(tokens)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(token_ids)

