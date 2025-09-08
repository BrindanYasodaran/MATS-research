import json
from pathlib import Path
import os
import heapq
from tqdm import tqdm
import re

import google.generativeai as genai
import random

genai.configure(api_key="AIzaSyDzfBiuo4gJ1Jk0DVo99mQTdxHDf0LDqSA")  # set env var before running
MODEL_NAME = "gemini-1.5-pro"  # or "gemini-1.5-flash" if you want it faster
GEN_CFG = {
    "temperature": 0.7,
    "max_output_tokens": 100,
    "response_mime_type": "text/plain",
}

def make_gemini_with_system(system_instruction: str):
    return genai.GenerativeModel(
        MODEL_NAME,
        generation_config=GEN_CFG,
        system_instruction=system_instruction
    )


def main():

    traits = ['comforting']
    elicited_dir = Path("/workspace/MATS-research/data/steer_llama_data").expanduser().resolve()

    for trait in traits:
        trait_file = elicited_dir / f"{trait}_elicited.json"
        print(f"Processing {trait_file}")
        with open(trait_file, "r") as f:

            trait_data = json.load(f)

            instr_pairs = trait_data["instruction"]
            evaluation_prompt = trait_data["eval prompt"]

            questions = trait_data["questions"]
            questions = questions[:20]

            results = {}


            for question in tqdm(questions):
                # ask pi for a response
                # pi_response = pi_model.generate(question)

                # # ask llama for a response
                # llama_response = llama_model.generate(question)

                # # ask qwen to judge the pi response using the eval prompt (0-100)
                # qwen_judge_of_pi_raw = qwen_model.generate(evaluation_prompt.format(question=question, answer=pi_response))
                # match_pi = re.search(r"\d+", str(qwen_judge_of_pi_raw))
                # qwen_judge_of_pi = int(match_pi.group()) if match_pi else 0

                # # ask qwen to judge the llama response using the eval prompt (0-100)
                # qwen_judge_of_llama_raw = qwen_model.generate(evaluation_prompt.format(question=question, answer=llama_response))
                # match_llama = re.search(r"\d+", str(qwen_judge_of_llama_raw))
                # qwen_judge_of_llama = int(match_llama.group()) if match_llama else 0

                # # take the difference in scores between pi and llama
                # score_difference = qwen_judge_of_pi - qwen_judge_of_llama

                # results[question] = {
                #     "pi_response": pi_response,
                #     "llama_response": llama_response,
                #     "score_difference": score_difference
                # }
                    # pick ONE of the five contrastive instruction pairs at random for this question
                # pair = random.choice(instr_pairs)
                pair = instr_pairs[1]
                sys_pos = pair["pos"]
                sys_neg = pair["neg"]

                # make two Gemini models with different system prompts
                model_pos = make_gemini_with_system(sys_pos)
                model_neg = make_gemini_with_system(sys_neg)

                # generate two contrastive responses to the SAME question
                resp_pos = model_pos.generate_content(question)
                resp_neg = model_neg.generate_content(question)

                results[question] = {
                    "pos_response": resp_pos.text,
                    "neg_response": resp_neg.text
                }

        # write the results to a file
        with open(trait_file.with_name(f"{trait_file.stem}_{trait}_results.json"), "w") as f:
            json.dump(results, f)

        # filter the traits by score difference; threshold is 40
        # filtered_traits = []
        # for question in results:
        #     if abs(results[question]["score_difference"]) > 40:
        #         filtered_traits.append(question)
        
        # # write the filtered traits to a file   
        # with open(trait_file.with_name(f"{trait_file.stem}_filtered.json"), "w") as f:
        #     json.dump(filtered_traits, f)


if __name__ == "__main__":
    main()