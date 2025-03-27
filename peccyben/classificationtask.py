#**********************************
#
# Classification Task Functions
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

s3 = boto3.client('s3')

from peccyben.utils import Ben_Res2S3, Cost_per_inf, plot_ben_graph, plot_ben_costperf, plot_topx_cp, LLM_Infer_Text
from peccyben.evals import calculate_accuracy, calculate_toxicity_1, calculate_bertscore


#--------- Classification function : unified ---------
def Class_Text(method,region,model_id,model_kwargs,prompt_template,transcript_text,all_class_text,cacheconf="default",latencyOpt="optimized"):

    prompt = prompt_template.format(conversation_text=transcript_text,category_list=all_class_text)
    
    llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token = LLM_Infer_Text(method,region,model_id,model_kwargs,prompt,cacheconf,latencyOpt)
        
    return llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token


#--------- Classification task evaluation  ---------
def Class_Eval(response_class_list, ref_class_list):
    
    acc = calculate_accuracy(response_class_list, ref_class_list)

    bert_p, bert_r, bert_f1 = calculate_bertscore(response_class_list, ref_class_list)

    tox1 = calculate_toxicity_1(response_class_list)
    
    return acc, bert_p, bert_r, bert_f1, tox1
    

#--------- Classification benchmarking  ---------    
def Class_Ben(method,region,model_id,model_kwargs,prompt_template,s3_bucket,file_name,BENCH_KEY,task_folder,cost_key,save_id,SLEEP_SEC,SAMPLE_LEN=-1,PP_TIME=1,cacheconf="default",latencyOpt="optimized"):

    # batch inference
    pp_list = []
    input_token_list = []
    elapsed_time_list = []
    output_token_list = []
    output_tpot_list = []
    cache_input_token_list = []
    cache_output_token_list = []

    response_text_list = []
    transcript_text_list = []
    ref_category_text_list = []

    acc_score_list = []
    bertp_score_list = []
    bertr_score_list = []
    bertf1_score_list = []
    tox_score_list = []
    
    cost_list = []
    
    try:
        # take input_file (csv)from S3 bucket and put to local ./data folder 
        s3 = boto3.resource('s3')
        
        file_path = './data/'+file_name
        s3.Bucket(s3_bucket).download_file(task_folder+'/input-data/'+file_name, file_path)
        
        # load data
        df_input = pd.read_csv(file_path)
            
        contact_id_list = df_input.contact_id.values.tolist() 
        transcript_list = df_input.transcript.values.tolist() 
        ref_category_list = df_input.ref_category.values.tolist()
        all_class_list = df_input.all_category.values.tolist()
    
        if SAMPLE_LEN < len(transcript_list):
            BATCH_NUM = SAMPLE_LEN
        else: 
            BATCH_NUM = len(transcript_list)
    
        for j in range(PP_TIME):
            for i in range(BATCH_NUM): 
                print(j,'-',i,end='|')  
                pp_list.append(str(j))
                llm_response_org, elapsed_time, input_token, output_token, output_tpot, cache_input_token, cache_output_token = Class_Text(method,
                                                                                                    region,
                                                                                                    model_id,
                                                                                    model_kwargs,
                                                                                    prompt_template,
                                                                                    transcript_list[i],
                                                                                    all_class_list[i],
                                                                                    cacheconf,latencyOpt)
                
                time.sleep(SLEEP_SEC)
                            
                if(len(llm_response_org)==0):
                    llm_response_org = "no response."
                    
                llm_response = ""
                if("<results>" in llm_response_org):
                    for item in llm_response_org.split("</results>"):
                        if ("<results>" in item):
                            llm_response = llm_response + item [ item.find("<results>")+len("<results>") : ]
                else:
                    llm_response = llm_response_org
                llm_response = llm_response.replace("\n","")
                print(llm_response)
        
                cost = Cost_per_inf(input_token, output_token, model_id,cost_key)
                
                input_token_list.append(input_token)
                elapsed_time_list.append(elapsed_time)
                output_token_list.append(output_token)
                output_tpot_list.append(output_tpot)
                cache_input_token_list.append(cache_input_token) 
                cache_output_token_list.append(cache_output_token)
        
                response_text_list.append(llm_response)
                transcript_text_list.append(transcript_list[i])
                ref_category_text_list.append(str(ref_category_list[i]))
                
                acc = calculate_accuracy([llm_response],[str(ref_category_list[i])])
                bert_p, bert_r, bert_f1 = calculate_bertscore([llm_response],[str(ref_category_list[i])])
                tox_1 = calculate_toxicity_1([llm_response])
                
                acc_score_list.append(acc)
                bertp_score_list.append(bert_p)
                bertr_score_list.append(bert_r)
                bertf1_score_list.append(bert_f1)
                tox_score_list.append(tox_1)
                
                cost_list.append(cost)     
    
        print("Evaluation completed, saving the results...",end=' ')
    
        # save output to csv ./results and upload to S3
        OUTPUT_FILE = task_folder+'_'+save_id.replace('/','_')+'_output.csv'
        df_output = pd.DataFrame() 
        df_output["test_round"] = pp_list
        df_output["transcript"] = transcript_text_list
        df_output["response_class"] = response_text_list
        df_output["ref_class"] = ref_category_text_list
        df_output["acc_score"] = acc_score_list
        df_output["bertp_score"] = bertp_score_list
        df_output["bertr_score"] = bertr_score_list
        df_output["bertf1_score"] = bertf1_score_list    
        df_output["tox_score"] = tox_score_list     
        df_output.to_csv('./results/'+OUTPUT_FILE, index=False)
        print("Result file saved.")
        
        Ben_Res2S3(s3_bucket,OUTPUT_FILE,BENCH_KEY,task_folder)
    
        # evaluation
        acc, bert_p, bert_r, bert_f1, tox1 = Class_Eval(response_text_list, ref_category_text_list)

    except Exception as e:
        print(f"\n\nAn error occurred: {e}. Please try again...")
    else:
        return np.sum(elapsed_time_list), np.sum(input_token_list), np.sum(output_token_list), np.sum(output_tpot_list), acc, bert_p, bert_r, bert_f1, tox1, np.sum(cost_list), np.sum(cache_input_token_list), np.sum(cache_output_token_list)


