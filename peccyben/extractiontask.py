#**********************************
#
# Entity Extraction Task Functions
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
from peccyben.evals import calculate_accuracy, calculate_toxicity_1, calculate_bertscore


#--------- Extraction function : unified ---------
def Extract_Text(method,region,model_id,model_kwargs,prompt_template,entity_name,doc_text,cacheconf="default",latencyOpt="optimized"):
    
    prompt = prompt_template.format(name=entity_name,document=doc_text)
    
    llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token = LLM_Infer_Text(method,region,model_id,model_kwargs,prompt,cacheconf,latencyOpt)
        
    return llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token

    

#--------- Extraction task evaluation  ---------
def Extract_Eval(response_text_list, document_text_list):

    acc = calculate_accuracy(response_text_list,document_text_list)    

    #r1,r2,r3,r4 = calculate_rouge(response_text_list,document_text_list)
    #print("Average Rouge-L_sum = ", r1)

    #avg_ss_score = calculate_semantic_sim_titan(response_text_list,document_text_list)

    #bert_p, bert_r, bert_f1 = calculate_bertscore(response_text_list,document_text_list)

    tox1 = calculate_toxicity_1(response_text_list)
    
    #return r1, bert_p, bert_r, bert_f1, tox1
    return acc, tox1
    

#--------- Get document for extraction  ---------    
def Extract_Get_Doc(s3_bucket,file_name,task_folder):
    
    # take input_file (csv)from S3 bucket and put to local ./data folder 
    s3 = boto3.resource('s3')
    
    docfile_path = './data/'+file_name
    s3.Bucket(s3_bucket).download_file(task_folder+'/input-data/'+file_name, docfile_path)
    
    return    
    
    
#--------- Extraction benchmarking  ---------        
def Extract_Ben(method,region,model_id,model_kwargs,prompt_template,document_text,s3_bucket,file_name,BENCH_KEY,task_folder,cost_key,save_id,SLEEP_SEC,SAMPLE_LEN=-1,PP_TIME=1,cacheconf="default",latencyOpt="optimized"):

    # batch inference
    pp_list = []
    input_token_list = []
    elapsed_time_list = []
    output_token_list = []
    output_tpot_list = []
    cache_read_input_token_list = []
    cache_write_input_token_list = []

    response_text_list = []
    reference_text_list = []
    name_text_list = []

    acc_score_list = []
    rouge_score_list = []
    #bertr_score_list = []
    #bertp_score_list = []
    #bertf1_score_list = []
    tox_score_list = []
    
    cost_list = []

    try:
        # take input_file (csv)from S3 bucket and put to local ./data folder 
        s3 = boto3.resource('s3')
        
        entityfile_path = './data/'+file_name
        s3.Bucket(s3_bucket).download_file(task_folder+'/input-data/'+file_name, entityfile_path)    
        
        # load question
        df_input = pd.read_csv(entityfile_path)
        name_list = df_input.name.values.tolist() 
        reference_list = df_input.reference.values.tolist()
    
        if SAMPLE_LEN < len(name_list):
            BATCH_NUM = SAMPLE_LEN
        else: 
            BATCH_NUM = len(name_list)
    
        for j in range(PP_TIME):
            for i in range(BATCH_NUM): 
                print(j,'-',i,end='|')  
                pp_list.append(str(j))
                llm_response_org, elapsed_time, input_token, output_token, output_tpot, cache_read_input_token, cache_write_input_token = Extract_Text(method,
                                                                                                        region,
                                                                                                        model_id,
                                                                                        model_kwargs,
                                                                                        prompt_template,
                                                                                        name_list[i],
                                                                                        document_text,
                                                                                    cacheconf,latencyOpt)
                time.sleep(SLEEP_SEC)
            
                if(len(llm_response_org)==0):
                    llm_response = "no response."
        
                llm_response = ""
                if("<results>" in llm_response_org):
                    for item in llm_response_org.split("</results>"):
                        if ("<results>" in item):
                            llm_response = llm_response + item [ item.find("<results>")+len("<results>") : ]
                else:
                    llm_response = llm_response_org
                llm_response = llm_response.replace("\n","")
                print(name_list[i],'-',llm_response)
                
                cost = Cost_per_inf(input_token, output_token, model_id,cost_key)
        
                input_token_list.append(input_token)
                elapsed_time_list.append(elapsed_time)
                output_token_list.append(output_token)
                output_tpot_list.append(output_tpot)
                cache_read_input_token_list.append(cache_read_input_token)
                cache_write_input_token_list.append(cache_write_input_token)
        
                response_text_list.append(llm_response)
                reference_text_list.append(reference_list[i])
                name_text_list.append(name_list[i])
                
                acc = calculate_accuracy([llm_response],[reference_list[i]])
                #r_1,r_2,r_3,r_4 = calculate_rouge([llm_response],[reference_list[i]])
                #bert_p, bert_r, bert_f1 = calculate_bertscore([llm_response],[reference_list[i]])
                tox_1 = calculate_toxicity_1([llm_response])
                
                acc_score_list.append(acc)
                #rouge_score_list.append(r_1)
                #bertr_score_list.append(bert_p)
                #bertp_score_list.append(bert_r)
                #bertf1_score_list.append(bert_f1)
                tox_score_list.append(tox_1)
                
                cost_list.append(cost)        
    
        print("Evaluation completed, saving the results...",end=' ')
    
        # save output to csv ./results and upload to S3
        OUTPUT_FILE = task_folder+'_'+save_id.replace('/','_')+'_output.csv'
        df_output = pd.DataFrame()  
        df_output["test_round"] = pp_list
        df_output["name_text"] = name_text_list
        df_output["reference_text"] = reference_text_list
        df_output["response_text"] = response_text_list
        #df_output["rouge_score"] = rouge_score_list
        df_output["acc_score"] = acc_score_list
        #df_output["bert_p_score"] = bertp_score_list
        #df_output["bert_r_score"] = bertr_score_list
        #df_output["bert_f1_score"] = bertf1_score_list
        df_output["tox_score"] = tox_score_list 
        df_output.to_csv('./results/'+OUTPUT_FILE, index=False)
    
        Ben_Res2S3(s3_bucket,OUTPUT_FILE,BENCH_KEY,task_folder)
    
        # evaluation
        acc, tox1 = Extract_Eval(response_text_list, reference_text_list)
    
    except Exception as e:
        print(f"\n\nAn error occurred: {e}. Please try again...")
    else:
        return np.sum(elapsed_time_list), np.sum(input_token_list), np.sum(output_token_list), np.sum(output_tpot_list), acc, tox1, np.sum(cost_list), np.sum(cache_read_input_token_list), np.sum(cache_write_input_token_list)


