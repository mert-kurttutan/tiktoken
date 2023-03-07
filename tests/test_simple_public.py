import tiktoken
import pytest
import time
import os

num_threads = int(os.environ["RAYON_NUM_THREADS"])

@pytest.fixture(scope="session")
def text_array() -> list[str]:
    text_arr = []
    with open("./data/big.txt", "r") as f:
        content = f.read()
        text_arr.append(content)

    return text_arr[:10000]*16
def test_rust_batch(benchmark, text_array):
    # Note that there are more actual tests, they're just not currently public :-)
    enc = tiktoken.get_encoding("gpt2")
    enc = tiktoken.encoding_for_model("text-davinci-003")
    result = benchmark(enc.encode_ordinary_batch_rust, text_array)


def test_python_batch(benchmark, text_array):
    # Note that there are more actual tests, they're just not currently public :-)
    enc = tiktoken.get_encoding("gpt2")
    enc = tiktoken.encoding_for_model("text-davinci-003")
    result = benchmark(enc.encode_ordinary_batch, text_array, num_threads=num_threads)




def test_simple():
    # Note that there are more actual tests, they're just not currently public :-)
    enc = tiktoken.get_encoding("gpt2")
    assert enc.encode("hello world") == [31373, 995]
    assert enc.decode([31373, 995]) == "hello world"
    assert enc.encode("hello <|endoftext|>", allowed_special="all") == [31373, 220, 50256]

    enc = tiktoken.get_encoding("cl100k_base")
    assert enc.encode("hello world") == [15339, 1917]
    assert enc.decode([15339, 1917]) == "hello world"
    assert enc.encode("hello <|endoftext|>", allowed_special="all") == [15339, 220, 100257]

    for enc_name in tiktoken.list_encoding_names():
        enc = tiktoken.get_encoding(enc_name)
        for token in range(10_000):
            assert enc.encode_single_token(enc.decode_single_token_bytes(token)) == token


def test_encoding_for_model():
    enc = tiktoken.encoding_for_model("gpt2")
    assert enc.name == "gpt2"
    enc = tiktoken.encoding_for_model("text-davinci-003")
    assert enc.name == "p50k_base"
