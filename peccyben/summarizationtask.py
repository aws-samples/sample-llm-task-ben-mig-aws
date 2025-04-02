#**********************************
#
# Summarization Task Functions
#
#**********************************

import json
import os
import sys
import time

import boto3
import botocore

import numpy as np
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

s3 = boto3.client('s3')

from peccyben.utils import Ben_Res2S3, Cost_per_inf, plot_ben_graph, plot_ben_costperf, plot_topx_cp, LLM_Infer_Text
from peccyben.evals import calculate_accuracy, calculate_toxicity_1, calculate_bertscore, calculate_rouge, calculate_semantic_sim_titan
from peccyben.evals import llm_judge_summ
from peccyben.utils import extract_strings_recursive


#--------- Summarization function : unified ---------
def Summ_Text(method,region,model_id,model_kwargs,prompt_template,doc_text,cacheconf="default",latencyOpt="optimized"):

    prompt = prompt_template.format(document=doc_text)
    
    llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token = LLM_Infer_Text(method,region,model_id,model_kwargs,prompt,cacheconf,latencyOpt)
        
    return llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token


    
#--------- Summarization task evaluation  ---------
def Summ_Eval(response_text_list, document_text_list):
    
    r1,r2,r3,r4 = calculate_rouge(response_text_list,document_text_list)

    avg_ss_score = calculate_semantic_sim_titan(response_text_list,document_text_list)

    bert_p, bert_r, bert_f1 = calculate_bertscore(response_text_list,document_text_list)

    tox1 = calculate_toxicity_1(response_text_list)
    
    return r1, avg_ss_score, bert_f1, tox1
    
    
#--------- Summarization benchmarking  ---------    
def Summ_Ben(method,region,model_id,jm1,jm2,model_kwargs,prompt_template,s3_bucket,file_name,BENCH_KEY,task_folder,cost_key,save_id,SLEEP_SEC,SAMPLE_LEN=-1,PP_TIME=1,cacheconf="default",latencyOpt="optimized"):

    # batch inference
    pp_list = []
    input_token_list = []
    elapsed_time_list = []
    output_token_list = []
    output_tpot_list = []
    cache_read_input_token_list = []
    cache_write_input_token_list = []

    response_text_list = []
    document_text_list = []

    rouge_score_list = []
    ss_score_list = []
    bertf1_score_list = []
    tox_score_list = []
    lj_score_list = []
    lj_thinking_list = []
    
    cost_list = []

    try:
        # take input_file (csv)from S3 bucket and put to local ./data folder 
        s3 = boto3.resource('s3')
        
        file_path = './data/'+file_name
        s3.Bucket(s3_bucket).download_file(task_folder+'/input-data/'+file_name, file_path)
        
        # load data
        df_input = pd.read_csv(file_path)
        section_id_list = df_input.Section_id.values.tolist() 
        section_title_list = df_input.Section_title.values.tolist() 
        section_text_list = df_input.Section_text.values.tolist()
        
        if SAMPLE_LEN < len(section_text_list):
            BATCH_NUM = SAMPLE_LEN
        else: 
            BATCH_NUM = len(section_text_list)
    
        for j in range(PP_TIME):        
            for i in range(BATCH_NUM): 
                print(j,'-',i,end='|')  
                pp_list.append(str(j))
                
                llm_response, elapsed_time, input_token, output_token, output_tpot, cache_read_input_token, cache_write_input_token = Summ_Text(method,region,
                                                                                               model_id,
                                                                                    model_kwargs,
                                                                                    prompt_template,
                                                                                    section_text_list[i],
                                                                                    cacheconf,latencyOpt)
                time.sleep(SLEEP_SEC)
                
                if(len(llm_response)==0):
                    llm_response = "no response."
                
                cost = Cost_per_inf(input_token, output_token, model_id, cost_key)
                
                input_token_list.append(input_token)
                elapsed_time_list.append(elapsed_time)
                output_token_list.append(output_token)
                output_tpot_list.append(output_tpot)
                cache_read_input_token_list.append(cache_read_input_token)
                cache_write_input_token_list.append(cache_write_input_token)
        
                response_text_list.append(llm_response)
                document_text_list.append(section_text_list[i])
        
                r_1,r_2,r_3,r_4 = calculate_rouge([llm_response],[section_text_list[i]])
                ss_score = calculate_semantic_sim_titan([llm_response],[section_text_list[i]])
                bert_p, bert_r, bert_f1 = calculate_bertscore([llm_response],[section_text_list[i]])
                tox_1 = calculate_toxicity_1([llm_response])
                
                rouge_score_list.append(r_1)
                ss_score_list.append(ss_score)
                bertf1_score_list.append(bert_f1)
                tox_score_list.append(tox_1)
                
                # add llm judge metrics
                lj_score, lj_thinking = llm_judge_summ(jm1,jm2,section_text_list[i],llm_response) 
                print("LLM_Judge_score=",lj_score)
                lj_score_list.append(lj_score)
                lj_thinking_list.append(lj_thinking)        
                 
                cost_list.append(cost)        
                
        print("Evaluation completed, saving the results...",end=' ')    
        
        # save output to csv ./results and upload to S3
        OUTPUT_FILE = task_folder+'_'+save_id.replace('/','_')+'_output.csv'
        df_output = pd.DataFrame() 
        df_output["test_round"] = pp_list        
        df_output["doc_text"] = document_text_list
        df_output["summ_text"] = response_text_list
        df_output["rouge_score"] = rouge_score_list
        df_output["bertf1_score"] = bertf1_score_list
        df_output["tox_score"] = tox_score_list     
        df_output["lj_score"] = lj_score_list 
        df_output["lj_thinking"] = lj_thinking_list 
        df_output.to_csv('./results/'+OUTPUT_FILE, index=False)
        
        Ben_Res2S3(s3_bucket,OUTPUT_FILE,BENCH_KEY,task_folder)
    
        # evaluation
        r1, avg_ss_score, bert_f1, tox1 = Summ_Eval(response_text_list, document_text_list)

    except Exception as e:
        print(f"\n\nAn error occurred: {e}. Please try again...")
    else:
        return np.sum(elapsed_time_list), np.sum(input_token_list), np.sum(output_token_list), np.sum(output_tpot_list), r1, avg_ss_score, bert_f1, np.average(lj_score_list), tox1, np.sum(cost_list), np.sum(cache_read_input_token_list), np.sum(cache_write_input_token_list) 
    

