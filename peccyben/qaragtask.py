#**********************************
#
# Question-Answer/RAG Task Functions
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

from peccyben.utils import Ben_Res2S3, Cost_per_inf, plot_ben_graph, plot_ben_costperf, plot_topx_cp, LLM_Infer_Text,Init_Bedrock
from peccyben.evals import calculate_accuracy, calculate_toxicity_1, calculate_bertscore, calculate_rouge, calculate_semantic_sim_titan

from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

from langchain_community.chat_models import BedrockChat

from datasets import Dataset
from ragas import evaluate
import nest_asyncio  # CHECK NOTES

from ragas.metrics import (
    answer_relevancy,
    answer_correctness,
    answer_similarity,
    context_precision,
    faithfulness,
    context_recall,
)

#from ragas.metrics.critique import harmfulness

# list of metrics we're going to use
metrics = [
    answer_correctness,
    answer_similarity,
    answer_relevancy,
    #faithfulness,
    context_recall,
    context_precision,
    #harmfulness,
]


def run_ragas(model_id,dataset):
    config = {
        "credentials_profile_name": "default",  # E.g "default"
        "region_name": "us-west-2",  # E.g. "us-east-1"
        "model_id": model_id,  # E.g "anthropic.claude-v2"
        "model_kwargs": {"temperature": 0.05},
    }

    bedrock_model = BedrockChat(
        #credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
        endpoint_url=f"https://bedrock-runtime.{config['region_name']}.amazonaws.com",
        model_id=config["model_id"],
        model_kwargs=config["model_kwargs"],
    )

    # init the embeddings
    bedrock_embeddings = BedrockEmbeddings(
        #credentials_profile_name=config["credentials_profile_name"],
        region_name=config["region_name"],
    )
    
    # NOTES: Only used when running on a jupyter notebook, otherwise comment or remove this function.
    nest_asyncio.apply()

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=bedrock_model,
        embeddings=bedrock_embeddings,
    )

    return result 


#--------- QA function: unified ---------
def QA_Text(method,region,model_id,model_kwargs,prompt_template,question_text,context_text,cacheconf="default",latencyOpt="optimized"):
    
    prompt = prompt_template.format(context=context_text, question=question_text)
    
    llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token = LLM_Infer_Text(method,region,model_id,model_kwargs,prompt,cacheconf,latencyOpt)
    
    return llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token


#--------- QA task evaluation  ---------
def QA_Eval(response_text_list, document_text_list):
    
    r1,r2,r3,r4 = calculate_rouge(response_text_list,document_text_list)

    avg_ss_score = calculate_semantic_sim_titan(response_text_list,document_text_list)

    bert_p, bert_r, bert_f1 = calculate_bertscore(response_text_list,document_text_list)

    tox1 = calculate_toxicity_1(response_text_list)
    
    return r1, avg_ss_score, bert_f1, tox1


#--------- QA create local vector db ---------
def QA_Create_VDB(chunk_size,chunk_overlap,embeddings,vdb_name,s3_bucket,file_name,task_folder):
        
    # take input_file (csv)from S3 bucket and put to local ./data folder 
    s3 = boto3.resource('s3')
    
    docfile_path = './data/'+file_name
    s3.Bucket(s3_bucket).download_file(task_folder+'/input-data/'+file_name, docfile_path)
    
    # create vdb 
    loader = PyPDFLoader(docfile_path)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,  # 1000
        chunk_overlap = chunk_overlap,  # 100
    )

    docs = text_splitter.split_documents(documents)

    vectorstore_faiss = FAISS.from_documents(
        docs,
        embeddings,
    )

    wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

    vectorstore_faiss.save_local(vdb_name)

    return 


#--------- QA benchmarking V2 (adding RAGAS) ---------    
def QA_Ben(method,region,model_id,judge_model_id,model_kwargs,prompt_template,vdb_name,s3_bucket,file_name,BENCH_KEY,task_folder,cost_key,save_id,SLEEP_SEC,SAMPLE_LEN=-1,PP_TIME=1,cacheconf="default",latencyOpt="optimized"):

    boto3_bedrock_runtime = Init_Bedrock(region)
    
    # batch inference
    pp_list = []
    input_token_list = []
    elapsed_time_list = []
    output_token_list = []
    output_tpot_list = []
    cache_input_token_list = []
    cache_output_token_list = []

    response_text_list = []
    context_text_list = []
    reference_text_list = []
    question_text_list = []

    rouge_score_list = []
    ss_score_list = []
    bertf1_score_list = []
    tox_score_list = []
    
    cost_list = []
    
    try:    
        # create embedding
        bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=boto3_bedrock_runtime)
        
        # load vectordb 
        vectorstore_faiss_general = FAISS.load_local(vdb_name, bedrock_embeddings,allow_dangerous_deserialization=True)
        wrapper_store_faiss_general = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss_general)
        
        # vectordb search
        SEARCH_K = 5
        retriever = vectorstore_faiss_general.as_retriever(search_kwargs={
            'k': SEARCH_K,
        })
    
        # take input_file (csv)from S3 bucket and put to local ./data folder 
        s3 = boto3.resource('s3')
        
        qfile_path = './data/'+file_name
        s3.Bucket(s3_bucket).download_file(task_folder+'/input-data/'+file_name, qfile_path)    
        
        # load question
        df_input = pd.read_csv(qfile_path)
        question_org_list = df_input.instruction.values.tolist() 
        reference_org_list = df_input.response.values.tolist()
        
        if SAMPLE_LEN < len(question_org_list):
            BATCH_NUM = SAMPLE_LEN
        else: 
            BATCH_NUM = len(question_org_list)
    
        for j in range(PP_TIME):
            for i in range(BATCH_NUM): 
                print(i,end='|')
                pp_list.append(str(j))
                print(question_org_list[i])
                
                docs = retriever.get_relevant_documents(question_org_list[i])
                #print(docs)
                
                contexts = []
                for k in range(len(docs)):
                    context_text = docs[k].page_content
                    #print(context_text)
                    contexts.append(context_text)
                    
                llm_response_org, elapsed_time, input_token, output_token, output_tpot, cache_input_token, cache_output_token = QA_Text(method,
                                                                                                 region,
                                                                                                 model_id,
                                                                                      model_kwargs,
                                                                                      prompt_template,
                                                                                      question_org_list[i],
                                                                                      docs,
                                                                                    cacheconf,latencyOpt)
                time.sleep(SLEEP_SEC)
        
                llm_response = ""
                if("<results>" in llm_response_org):
                    for item in llm_response_org.split("</results>"):
                        if ("<results>" in item):
                            llm_response = llm_response + item [ item.find("<results>")+len("<results>") : ]
                else:
                    llm_response = llm_response_org
                llm_response = llm_response.replace("\n","")
                print(llm_response)
                
                cost = Cost_per_inf(input_token, output_token, model_id, cost_key)
                
                input_token_list.append(input_token)
                elapsed_time_list.append(elapsed_time)
                output_token_list.append(output_token)
                output_tpot_list.append(output_tpot)
        
                response_text_list.append(llm_response)
                question_text_list.append(question_org_list[i])
                reference_text_list.append(reference_org_list[i])
                context_text_list.append(contexts)
                
                r_1,r_2,r_3,r_4 = calculate_rouge([llm_response],[reference_org_list[i]])
                ss_score = calculate_semantic_sim_titan([llm_response],[reference_org_list[i]])
                tox_1 = calculate_toxicity_1([llm_response])
                
                rouge_score_list.append(r_1)
                ss_score_list.append(ss_score)
                tox_score_list.append(tox_1)
                
                cost_list.append(cost)        
    
        print("Evaluation completed, saving the results...",end=' ')
    
        # save output to csv ./results and upload to S3
        OUTPUT_FILE = task_folder+'_'+save_id.replace('/','_')+'_output.csv'
        df_output = pd.DataFrame()  
        df_output["question_text"] = question_text_list
        df_output["reference_text"] = reference_text_list
        df_output["context_text"] = context_text_list
        df_output["response_text"] = response_text_list
        df_output["rouge_score"] = rouge_score_list
        df_output["ss_score"] = ss_score_list
        df_output["tox_score"] = tox_score_list 
            
        r1, avg_ss_score, bert_f1, tox1 = QA_Eval(response_text_list, reference_text_list)
        
        # Run RAGAS
        ds2 = pd.DataFrame()
        ds2["question"] = question_text_list
        ds2["ground_truth"] = reference_text_list
        ds2["answer"] = response_text_list
        ds2["contexts"] = context_text_list
        ds2["retreived_contexts"] = context_text_list
        
        ds2_dataset = Dataset.from_pandas(ds2)
        
        ragas_result = run_ragas(judge_model_id,ds2_dataset)
        
        df_ragas = ragas_result.to_pandas()
        
        # Merge and save
        df_output = pd.concat([df_output, df_ragas], axis=1)
        df_output.to_csv('./results/'+OUTPUT_FILE, index=False)
        Ben_Res2S3(s3_bucket,OUTPUT_FILE,BENCH_KEY,task_folder)        

    except Exception as e:
        print(f"\n\nAn error occurred: {e}. Please try again...")
    else:
        return np.sum(elapsed_time_list), np.sum(input_token_list), np.sum(output_token_list), np.sum(output_tpot_list), r1, avg_ss_score, bert_f1, tox1, np.mean(ragas_result['answer_correctness']), np.mean(ragas_result['semantic_similarity']), np.mean(ragas_result['answer_relevancy']), np.mean(ragas_result['context_recall']), np.mean(ragas_result['context_precision']), np.sum(cost_list), np.sum(cache_input_token_list), np.sum(cache_output_token_list)



