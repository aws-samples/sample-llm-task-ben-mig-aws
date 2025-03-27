#**********************************
#
# Prompt Evaluation Task Functions
#
#**********************************

import json
import os
import sys
import time
import math

import boto3
import botocore

import numpy as np
import pandas as pd

s3 = boto3.client('s3')



from peccyben.utils import Ben_Res2S3, Cost_per_inf, plot_ben_graph, plot_ben_costperf, plot_topx_cp, LLM_Infer_Text
from peccyben.evals import calculate_accuracy, calculate_toxicity_1, calculate_bertscore, calculate_rouge, calculate_semantic_sim_titan


#--------- PromptEval function: unified ---------
def PromptEval_Text(method,region,model_id,model_kwargs,prompt_template,inputs,cacheconf="default",latencyOpt="optimized"):

    prompt = prompt_template.format(inputs=inputs)
    
    llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token = LLM_Infer_Text(method,region,model_id,model_kwargs,prompt,cacheconf,latencyOpt)        

    
    return llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token


#--------- PromptEval benchmarking  ---------    
def PromptEval_Ben(method,region,model_id,model_kwargs,prompt_template,s3_bucket,file_name,BENCH_KEY,task_folder,cost_key,save_id,SLEEP_SEC,SAMPLE_LEN=-1,PP_TIME=1,cacheconf="default",latencyOpt="optimized"):

    # batch inference
    pp_list = []
    input_token_list = []
    elapsed_time_list = []
    output_token_list = []
    output_tpot_list = []
    cache_input_token_list = []
    cache_output_token_list = []

    response_text_list = []
    prompt_text_list = []
    
    cost_list = []

    try:
    
        # take input_file (csv)from S3 bucket and put to local ./data folder 
        s3 = boto3.resource('s3')
        
        file_path = './data/'+file_name
        s3.Bucket(s3_bucket).download_file(task_folder+'/input-data/'+file_name, file_path)
        
        # load data
        df_input = pd.read_csv(file_path)
        
        inputs_list = df_input["prompt_inputs"].tolist() 
        
        if SAMPLE_LEN < len(inputs_list):
            BATCH_NUM = SAMPLE_LEN
        else: 
            BATCH_NUM = len(inputs_list)
            
        for j in range(PP_TIME):    
            for i in range(BATCH_NUM): 
                print(i,end='|')
                pp_list.append(str(j))                
                llm_response, elapsed_time, input_token, output_token, output_tpot, cache_input_token, cache_output_token = PromptEval_Text(method,
                                                                                                        region,
                                                                                                        model_id,
                                                                                            model_kwargs,
                                                                                            prompt_template,
                                                                                            inputs_list[i],
                                                                                        cacheconf,latencyOpt)
    
                time.sleep(SLEEP_SEC)
    
                if(len(llm_response)==0):
                    llm_response = "no response."
                    
                cost = Cost_per_inf(input_token, output_token, model_id, cost_key)
    
                input_token_list.append(input_token)
                elapsed_time_list.append(elapsed_time)
                output_token_list.append(output_token)
                output_tpot_list.append(output_tpot)
                cache_input_token_list.append(cache_input_token) 
                cache_output_token_list.append(cache_output_token)
    
                response_text_list.append(llm_response)
                prompt = prompt_template.format(inputs=inputs_list[i])
                prompt_text_list.append(prompt)
    
                cost_list.append(cost)        
    
        print("Evaluation completed, saving the results...",end=' ')
        
        # save output to csv ./results and upload to S3
        OUTPUT_FILE = task_folder+'_'+save_id.replace('/','_')+'_output.csv'
        df_output = pd.DataFrame() 
        df_output["test_round"] = pp_list
        df_output["prompt"] = prompt_text_list
        df_output["response"] = response_text_list
        df_output["input_token"] = input_token_list
        df_output["output_token"] = output_token_list
        df_output["elapsed_time"] = elapsed_time_list
        df_output["cache_input_token"] = cache_input_token_list
        df_output["cache_output_token"] = cache_output_token_list
        
        df_output.to_csv('./results/'+OUTPUT_FILE, index=False)
        
        Ben_Res2S3(s3_bucket,OUTPUT_FILE,BENCH_KEY,task_folder)
        
        return_list = []
        return_list.append(np.sum(elapsed_time_list))
        return_list.append(np.sum(input_token_list))
        return_list.append(np.sum(output_token_list))
        return_list.append(np.sum(output_tpot_list)) 
        return_list.append(np.sum(cost_list))
        return_list.append(np.sum(cache_input_token_list))
        return_list.append(np.sum(cache_output_token_list))

    except Exception as e:
        print(f"\n\nAn error occurred: {e}. Please try again...")
    else:
        return return_list





