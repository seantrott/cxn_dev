"""Runs Pythia models on DAIS dataset."""


## Reference code: https://github.com/taka-yamakoshi/neural_constructions/blob/master/DAIS/analysis/CalcSentProbs.py

import pandas as pd
import torch
import pandas as pd
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from tqdm import tqdm
import utils
import os
import math


### Models to test
MODELS = [
         'EleutherAI/pythia-14m',
         'EleutherAI/pythia-70m',
         'EleutherAI/pythia-160m',
         'EleutherAI/pythia-410m'
          ]


def calculate_sentence_surprisal(sentence, model, tokenizer, device):
    """
    Computes total and mean token-level surprisal (in bits) for a sentence.
    Surprisal = -log2 P(token_i | context)

    Returns:
      total_surprisal (float): sum of surprisal over tokens
      mean_surprisal (float): average surprisal per token
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        outputs = model(**inputs, labels=input_ids)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
    # Calculate log probabilities over vocabulary for each token
    log_probs = torch.log_softmax(logits, dim=-1)  # log probs in natural log (nats)
    
    # Shift input_ids to get target tokens (predict token i given tokens < i)
    target_ids = input_ids[:, 1:]
    # Align log_probs to predict next token: ignore the first token prediction
    log_probs = log_probs[:, :-1, :]
    
    # Gather log probs corresponding to the actual next token
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # shape: [1, seq_len-1]
    
    # Convert from nats to bits: divide by ln(2)
    token_surprisals = -token_log_probs / math.log(2)  # positive surprisal
    
    # Sum and mean surprisal
    total_surprisal = token_surprisals.sum().item()
    mean_surprisal = token_surprisals.mean().item()
    
    return total_surprisal, mean_surprisal


### Handle logic for a dataset/model
def main(df, mpath, revisions):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("number of checkpoints:", len(revisions))

    for checkpoint in tqdm(revisions):

        for seed in range(1, 10):

            seed_name = "seed" + str(seed)

            ### Set up save path, filename, etc.
            savepath = "data/processed/dais_rs/"
            if not os.path.exists(savepath): 
                os.mkdir(savepath)
            if "/" in mpath:
                filename = "dais-surprisals-rs_model-" + mpath.split("/")[1] + "-" + checkpoint + "-" + seed_name + ".csv"
            else:
                filename = "dais-surprisals-rs_model-" + mpath +  "-" + checkpoint + "-" + seed_name + ".csv"

            print("Checking if we've already run this analysis...")
            print(filename)
            if os.path.exists(os.path.join(savepath,filename)):
                print("Already run this model for this checkpoint.")
                continue

            model_name = mpath + "-" + seed_name
            print(model_name)

            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision=checkpoint,
                output_hidden_states = True
            )
            model.to(device) # allocate model to desired device

            tokenizer = AutoTokenizer.from_pretrained(mpath, revision=checkpoint)


            n_layers = model.config.num_hidden_layers
            print("number of layers:", n_layers)
        
            n_params = utils.count_parameters(model)
        
            results = []

            for (ix, row) in tqdm(df.iterrows(), total=df.shape[0]):

                do_sent = str(row["DOsentence"]) 
                pd_sent = str(row["PDsentence"]) 

                do_surprisal_total, do_surprisal_mean = calculate_sentence_surprisal(do_sent, model, tokenizer, device)
                pd_surprisal_total, pd_surprisal_mean = calculate_sentence_surprisal(pd_sent, model, tokenizer, device)


                results.append({
                    'do_surprisal_total': do_surprisal_total,
                    'pd_surprisal_total': pd_surprisal_total,
                    'do_surprisal_mean': do_surprisal_mean,
                    'pd_surprisal_mean': pd_surprisal_mean,
                    'do_sent': do_sent,
                    'pd_sent': pd_sent,
                    'log_odds_total': pd_surprisal_total - do_surprisal_total,
                    'log_odds_mean': pd_surprisal_mean - do_surprisal_mean,
                    'classification': row['classification'],
                    'BehavDOpreference': row['BehavDOpreference']
                    
                })


            df_results = pd.DataFrame(results)
            df_results['n_params'] = n_params
            df_results['mpath'] = mpath
            df_results['revision'] = checkpoint
            df_results['seed_name'] = seed_name
            df_results['seed'] = seed
            df_results['step'] = int(checkpoint.replace("step", ""))
            
            
            ### Hurray! Save your cosine distance results to load into R
            #.  for analysis
        
        
            df_results.to_csv(os.path.join(savepath,filename), index=False)



if __name__ == "__main__":

    ### Read in dataset
    df_judgments = pd.read_csv("data/raw/DAIS/generated_pairs_with_results.csv")

    ### Get revisions
    revisions = utils.generate_revisions_test()

    ## Run main
    for mpath in MODELS:
        main(df_judgments, mpath, revisions)
