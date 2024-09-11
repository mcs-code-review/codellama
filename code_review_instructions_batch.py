import json
import math
import os
import re
from typing import Optional

import fire
import pandas as pd
import torch
from evaluation import myeval
from llama import Llama
from tqdm import tqdm


class Config:
    def __init__(self, conf_path):
        """
        conf_path: a json file storing configs
        """
        with open(conf_path, "r") as json_file:
            conf = json.load(json_file)

        for key, value in conf.items():
            setattr(self, key, value)


def save_output(cfg, df):
    dataset_name = os.path.splitext(os.path.basename(cfg.in_path))[0]
    output_dir = f"{cfg.out_dir}/{cfg.model}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f"{cfg.out_dir}/{cfg.model}/{dataset_name}.jsonl"

    df.to_json(output_path, orient="records", lines=True)
    return output_path


def split_dataframe(df, chunk_size=1000):
    """
    Splits the DataFrame into smaller chunks
    df: a Dataframe, chunkSize: the chunk size
    return a list of DataFrames
    """
    chunks = list()
    num_chunks = math.ceil(len(df) / chunk_size)
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size : (i + 1) * chunk_size])
    return chunks


################################################# Main #################################################
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    conf_path: str,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_seq_len: int = 2048,
    max_batch_size: int = 8,
    max_gen_len: int = None,
    debug: bool = False,
):
    cfg = Config(conf_path)
    if debug:
        print(f"Config: {cfg.__dict__}")

    if torch.cuda.is_available():
        print(f"CUDA is available")
    else:
        print("CUDA is not available")
        return

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    def make_dialog(user_prompt):
        return [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    df = pd.read_json(path_or_buf=cfg.in_path, lines=True)
    proccessed_df = pd.DataFrame()

    for chunk in tqdm(
        split_dataframe(df, chunk_size=max_batch_size), desc="Processing"
    ):
        prompts = chunk.user_prompt.apply(make_dialog)

        if debug:
            print(f"Prompts: {len(df.index)}")

        results = generator.chat_completion(
            dialogs=prompts,
            temperature=temperature,
            top_p=top_p,
            max_gen_len=max_gen_len,
        )

        answers = [result["generation"]["content"] for result in results]
        proccessed = pd.DataFrame(answers, columns=["codellama_answer"])
        proccessed = pd.concat([chunk.reset_index(drop=True), proccessed], axis=1)
        proccessed["codellama_code"] = proccessed.codellama_answer.apply(
            extract_code_diff
        )

        proccessed_df = pd.concat([proccessed_df, proccessed])

    output_path = save_output(cfg, proccessed_df)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
