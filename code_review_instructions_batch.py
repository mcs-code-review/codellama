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


def remove_minus(code):
    """
    Remove the minus sign from the beginning of each line in the code.
    """
    return "\n".join([line[1:] for line in code.split("\n")])


def remove_plus(code):
    """
    Remove the plus sign from the beginning of each line in the code.
    """
    return "\n".join(
        [line[1:].strip() for line in code.split("\n") if line.strip() != ""]
    )


def create_prompt(comment, code_diff):
    remove_minus_code_diff = remove_minus(code_diff)
    user_prompt = f"""
    As a developer, imagine you've submitted a pull request and your team leader
    requests you to make a change to a piece of code. The old code being
    referred to in the hunk of code changes is:
    ```
    {remove_minus_code_diff}
    ```
    There is the code review for this code:
    {comment}
    Please generate the revised code according to the review
    """
    return user_prompt


def get_user_prompts(in_path):
    df = pd.read_json(path_or_buf=in_path, lines=True)
    df["user_prompt"] = df.apply(lambda x: create_prompt(x.review, x.old), axis=1)
    return df


def extract_code_diff(text):
    """
    Extract code diff from text. Code is assumed to be in the format:
    ```lang
    code
    ```
    where lang is the language of the code.
    """

    code = re.findall(r"```[A-Za-z]*\n(.*?)\n```", text, re.DOTALL)
    if code:
        return code[0]
    return "NO CODE"


def evaluate_code_diff(actual_code, refined_code):
    remove_plus_code_diff = remove_plus(actual_code)
    em, em_trim, _, _, bleu, bleu_trim = myeval(remove_plus_code_diff, refined_code)
    return em, em_trim, bleu, bleu_trim


def save_output(cfg, df):
    dataset_name = os.path.splitext(os.path.basename(cfg.in_path))[0]
    output_dir = f"{cfg.out_dir}/{cfg.model}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f"{cfg.out_dir}/{cfg.model}/{dataset_name}_output.jsonl"

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

    df = get_user_prompts(cfg.in_path)
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

        (
            proccessed["codellama_em"],
            proccessed["codellama_em_trim"],
            proccessed["codellama_bleu"],
            proccessed["codellama_bleu_trim"],
        ) = zip(
            *proccessed.apply(
                lambda row: evaluate_code_diff(row["new"], row["codellama_code"]),
                axis=1,
            )
        )

        proccessed_df = pd.concat([proccessed_df, proccessed])

    output_path = save_output(cfg, proccessed_df)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
