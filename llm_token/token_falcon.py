import sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "model_file_path",
    local_files_only=True,
    trust_remote_code=True)

token_ids = tokenizer.encode(sys.argv[1])
print(token_ids)
for tid in token_ids:
    res = tokenizer.decode([tid])
    print(res)

