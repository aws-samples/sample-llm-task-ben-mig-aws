#**********************************
#
# Common Functions
#
#**********************************

import pandas as pd
import json
import time

#from transformers import AutoModelForCausalLM, AutoTokenizer

import boto3

s3 = boto3.client('s3')


#--------- Init bedrock ---------
def Init_Bedrock(REGION):

    #boto3_bedrock = boto3.client(service_name="bedrock", region_name=REGION)
    boto3_bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=REGION)

    return boto3_bedrock_runtime


#--------- Upload benchmark results to S3 ---------
def Ben_Res2S3(S3_BUCKET,file_name,BENCH_KEY,task_folder):
    s3.upload_file('./results/'+file_name, 
                   S3_BUCKET, 
                   task_folder+'/output-results/'+BENCH_KEY+'/'+file_name)
    return 
    
    
    
#--------- Bedrock cost calculation ---------    
def Cost_per_inf(input_token, output_token, model_id, price_file):

    df_price = pd.read_csv(price_file)
    
    if model_id in df_price.model.values.tolist():
        COST_PER_1K_INPUT_TOKENS = df_price['input_token'][df_price['model']==model_id].to_list()[0]
        COST_PER_1K_OUTPUT_TOKENS = df_price['output_token'][df_price['model']==model_id].to_list()[0] 
    else:
        COST_PER_1K_INPUT_TOKENS = 0 
        COST_PER_1K_OUTPUT_TOKENS = 0 
    
    cost_per_inf = COST_PER_1K_INPUT_TOKENS*input_token/1000000+COST_PER_1K_OUTPUT_TOKENS*output_token/1000000
    
    return cost_per_inf 


#--------- Generate plots ---------  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


# Plot metrics graph
def plot_ben_graph(metric_idx,df_results,title,task_folder):
    fig, ax = plt.subplots()

    llm_list = df_results.columns.tolist()
    bar_labels = df_results.columns.tolist()
    #bar_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    bar_colors = []
    for name in mcolors.CSS4_COLORS:
        bar_colors.append(name)

    idx_num = metric_idx
    idx_name = df_results.index.tolist()[idx_num]

    ben_metrics = df_results.loc[idx_name]

    bar_c = []
    for i in range(0,len(llm_list)):
        bar_c.append(bar_colors[i+10])
        
    ax.bar(llm_list, ben_metrics, label=bar_labels, color=bar_c)
    ax.set_ylabel(idx_name)
    ax.set_xticklabels(llm_list,rotation = 90)

    #ax.set_title('AGI LLM Benchmark: Classification - ' + idx_name)
    ax.set_title(title + ' - ' + idx_name)
    ax.legend(loc=(1.04,0.5), fontsize="8")
    
    save_file_name = task_folder+'_'+idx_name+'.png'
    plt.savefig('./results/'+save_file_name, bbox_inches='tight')
    plt.show()
    
    return save_file_name


# Plot cost-perf graph
def plot_ben_costperf(df_results,perf_metric,title,task_folder):

    fig, ax = plt.subplots()

    llm_list = df_results.columns.tolist()
    #dot_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    dot_colors = []
    for name in mcolors.CSS4_COLORS:
        dot_colors.append(name)
    
    dot_c = []
    for i in range(0,len(llm_list)):
        dot_c.append(dot_colors[i+10])
        
    #x = df_results.loc['BERT-F1']
    x = df_results.loc[perf_metric]
    y = df_results.loc['Cost']

    ax.scatter(x=x, y=y, marker='o', s=80, c=dot_c, label=llm_list) 
    #for i, txt in enumerate(llm_list):
    #    ax.annotate(txt, (x[i]*1.005, y[i]*1.005), fontsize="8")

    #ax.set_xlabel('Accuracy (BERT-F1)')
    ax.set_xlabel('Performance ('+perf_metric+')')
    ax.set_ylabel('Cost (USD)')
    #ax.set_title('AGI LLM Benchmark: Classification - Cost-Performance Analysis')
    ax.set_title(title + ' - Cost-Performance Analysis')

    recs = []
    for i in range(0,len(llm_list)):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=dot_colors[i+10]))

    ax.legend(recs,llm_list,loc=(1.04,0.5),fontsize="8",title="LLM")

    plt.rc('grid', linestyle="--", color='black')
    plt.grid(True)
    
    save_file_name = task_folder+'_'+'cost_perf.png'
    plt.savefig('./results/'+save_file_name, bbox_inches='tight')
    plt.show()
    
    return save_file_name


# Plot cost-perf graph - topx
def plot_topx_cp(df,perf_metric,top_num,task_folder):
    df_cp = df
        
    df_cp.loc['cost_perf'] = df_cp.loc[perf_metric] / df_cp.loc['Cost']
    df_trans = df_cp.transpose()
    
    print("=== df_cp ===\n",df_trans)
    
    df_trans['cost_perf'] = pd.to_numeric(df_trans['cost_perf'], errors='coerce')

    topx = df_trans.nlargest(top_num, 'cost_perf')
    
    save_file_name = plot_ben_costperf(df_cp[topx.index],perf_metric,'Top '+str(top_num),task_folder)
    
    print(df_cp[topx.index].loc['cost_perf'])
    
    return save_file_name


#--------- JSON processing ---------  
def extract_strings_recursive(test_str, tag):
    # finding the index of the first occurrence of the opening tag
    start_idx = test_str.find("<" + tag + ">")
 
    # base case
    if start_idx == -1:
        return []
 
    # extracting the string between the opening and closing tags
    end_idx = test_str.find("</" + tag + ">", start_idx)
    res = [test_str[start_idx+len(tag)+2:end_idx]]
 
    # recursive call to extract strings after the current tag
    res += extract_strings_recursive(test_str[end_idx+len(tag)+3:], tag)
 
    return res


#-------- LLM inference: text tasks --------
def LLM_Infer_Text(method,region,model_id,model_kwargs,prompt,cacheconf="default",latencyOpt="optimized"):

    cache_input_token = 0
    cache_output_token = 0
    
    if(method=="HF"):
        llm_response, elapsed_time, input_token, output_token, throuput = LLM_Text_HF_Infer(model_id,model_kwargs,prompt)
    elif(method=="OLM"):
        llm_response, elapsed_time, input_token, output_token, throuput = LLM_Text_OLM_Infer(model_id,model_kwargs,prompt)
    elif(method=="Bedrock"):
        llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token = LLM_Text_Bedrock_Conv_Infer(region,model_id,model_kwargs,prompt,cacheconf,latencyOpt)
    else: # default is Bedrock
        llm_response, elapsed_time, input_token, output_token, throuput = LLM_Text_Bedrock_Infer(region,model_id,model_kwargs,prompt)

    return llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token


#-------- LLM inference by Bedrock (converse api) --------
def LLM_Text_Bedrock_Conv_Infer(region,model_id,model_kwargs,prompt,cacheconf="default",latencyOpt="optimized"):

    boto3_bedrock_runtime = Init_Bedrock(region)
    
    #system_prompts = [{"text": "Answer with \"Answer:\""}]   # some models do not support this
    messages = [] 
    if cacheconf=="default":
        messages.append({
            "role": "user",
            "content": [{"text": prompt,"cachePoint": {"type": cacheconf}}]
        })
    else:
        messages.append({
            "role": "user",
            "content": [{"text": prompt,}]
        })
        

    st = time.time()

    if latencyOpt=="optimized":
        response = boto3_bedrock_runtime.converse(
            modelId = model_id,
            messages = messages,
            #system = system_prompts,
            inferenceConfig = model_kwargs,
            #additionalModelRequestFields = {}      # do not use this option
            performanceConfig={'latency' : latencyOpt}
        )
    else:
        response = boto3_bedrock_runtime.converse(
            modelId = model_id,
            messages = messages,
            #system = system_prompts,
            inferenceConfig = model_kwargs,
            #additionalModelRequestFields = {}      # do not use this option
        )
        

    et = time.time()
    elapsed_time = et - st

    llm_response = response['output']['message']['content'][0]['text']
    
    token_usage = response['usage']
    input_token = token_usage['inputTokens']
    output_token = token_usage['outputTokens']
    #cache_input_token = token_usage["cacheReadInputTokenCount"]
    #cache_input_token = token_usage["cacheWriteInputTokenCount"]
    cache_input_token = 0
    cache_output_token = 0

    throuput = output_token/elapsed_time

    return llm_response, elapsed_time, input_token, output_token, throuput, cache_input_token, cache_output_token


#-------- LLM inference by Bedrock --------
def LLM_Text_Bedrock_Infer(region,model_id,model_kwargs,prompt):

    boto3_bedrock_runtime = Init_Bedrock(region)
    
    if ('titan' in model_id):    
        model_body = {
            "inputText": f"{prompt}"
        }
        model_body["textGenerationConfig"] =  model_kwargs  
    elif ('olympus' in model_id or 'nova' in model_id):
        model_body = {
          "schemaVersion": "messages-v1",
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "text": prompt
                }
              ]
            }
          ],
          "system": [
            {
              "type": "system",
              "content": [
                {
                  "text": "Answer with \"Answer:\""
                }
              ]
            },
          ],
        }
        model_body["inferenceConfig"] =  model_kwargs
    elif ('claude-3' in model_id):
        model_body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1024,
                        "top_p": 0.9,
                        "temperature": 0,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}],
                            }
                        ],
        }
    else:
        model_body = {
            "prompt": f"{prompt}"
        }
        model_body.update(model_kwargs)
    
    body_bytes = json.dumps(model_body).encode('utf-8')

    st = time.time()    
    response = boto3_bedrock_runtime.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="*/*",
                body=body_bytes,
            )
    et = time.time()
    elapsed_time = et - st
    
    if ('titan' in model_id):
        response_body_json = json.loads(response['body'].read().decode('utf-8'))
        llm_response = response_body_json["results"][0]["outputText"].strip()
        llm_latency = response["ResponseMetadata"]["HTTPHeaders"]["x-amzn-bedrock-invocation-latency"]
    elif ('llama' in model_id):
        response_body_json = json.loads(response['body'].read().decode('utf-8'))
        llm_response = response_body_json["generation"].strip()
    elif ('claude-v2' in model_id or 'claude-instant-v1' in model_id ):
        response_body_json = json.loads(response['body'].read().decode('utf-8'))
        llm_response = response_body_json["completion"].strip()
    elif ('claude-3' in model_id):
        response_body_json = json.loads(response['body'].read().decode('utf-8'))
        llm_response = response_body_json["content"][0]["text"].strip()
    elif ('mistral' in model_id):
        response_body_json = json.loads(response['body'].read().decode('utf-8'))
        llm_response = response_body_json["outputs"][0]["text"].strip()    
    elif ('olympus' in model_id or 'nova' in model_id):
        response_body_json = json.loads(response['body'].read())
        llm_response = response_body_json['output']['message']['content'][0]['text'].strip()
    else :
        response_body_json = ""
        llm_response = 'MODEL TYPE NOT YET SUPPORTED.'

    if "usage" in response_body_json:
        if ('claude-3' in model_id):
            input_token = response_body_json['usage']['input_tokens']
            output_token = response_body_json['usage']['output_tokens']
        else:
            input_token = response_body_json['usage']['inputTokens']
            output_token = response_body_json['usage']['outputTokens']
    else:
        input_token = len(prompt.split())/0.75
        output_token = len(llm_response.split())/0.75
    
    throuput = output_token/elapsed_time
    
    return llm_response, elapsed_time, input_token, output_token, throuput



#-------- LLM inference by HF --------
def LLM_Text_HF_Infer(model_id,model_kwargs,prompt):

    input_token = len(prompt.split())/0.75

    device = "cuda"

    if ('titan' in model_id):
        text = f"{prompt}"
    elif ('llama' in model_id):
        text = f"{prompt}"
    elif ('mistral' in model_id):
        text = f"<s>[INST] {prompt} [/INST]" 
    elif ('olympus' in model_id or 'nova' in model_id):
        text = f"{prompt}"
    else :
        text = f"{prompt}"    


    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token = model_kwargs['model_token'],
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(      
        model_id, 
        use_cache=True,
        device_map="auto",
        token = model_kwargs['model_token'],
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))

    
    inputs = tokenizer(text, return_tensors="pt").to(device)

    st = time.time()
    
    outputs = model.generate(**inputs, 
                                do_sample=True,
                                max_new_tokens=4096, 
                                temperature=model_kwargs['temperature'], 
                                #top_k=top_k, 
                                top_p=model_kwargs['top_p'],
                                num_return_sequences=1,
                                repetition_penalty=1.1,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                use_cache=True,
                               )
    
    et = time.time()
    elapsed_time = et - st
    
    llm_response = (tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    output_token = len(llm_response.split())/0.75

    throuput = output_token/elapsed_time
    
    return llm_response, elapsed_time, input_token, output_token, throuput
    


#-------- LLM inference by ollama --------
def LLM_Text_OLM_Infer(model_id,model_kwargs,prompt):

    input_token = len(prompt.split())/0.75

    st = time.time()
    
    response: ChatResponse = chat(model=model_id, messages=[
      {
        'role': 'user',
        'content': prompt,
        'options': model_kwargs, 
      },
    ])
    
    et = time.time()
    elapsed_time = et - st
    
    llm_response = response['message']['content']
    llm_response = llm_response.split('</think>')[1]
    
    output_token = len(llm_response.split())/0.75

    throuput = output_token/elapsed_time
    
    return llm_response, elapsed_time, input_token, output_token, throuput   



#--------- Save benchmark results ---------    
def Ben_Save(Results_class,s3_bucket,bench_key,task_folder,perf_metric,top_x,TITLE):
    
    # Save metrics results
    OUTPUT_FILE = './results/'+task_folder+'_results.csv'
    Results_class_reset = Results_class.reset_index()
    Results_class_reset.to_csv(OUTPUT_FILE, index=False)
    
    # Save metrics plots
    for i in range(len(Results_class)):
        plot_ben_graph(i,Results_class,TITLE,task_folder)
        
    # cost-performance plot
    cp_file_name = plot_ben_costperf(Results_class,perf_metric,TITLE+' - ',task_folder)

    top_cp_file_name = plot_topx_cp(Results_class,perf_metric,top_x,task_folder)

    # Upload metrics data to S3
    print("\nUploading benchmark metrics ...")
    METRICS_RESULTS_FILENAME = task_folder+'_results.csv'
    Ben_Res2S3(s3_bucket,METRICS_RESULTS_FILENAME,bench_key,task_folder)

    # Upload metrics plots to S3
    metrics_list=list(Results_class.index)            # ['Inference_Time','Input_Token','Output_Token','Throughput','Accuracy','BERT-P','BERT-R','BERT-F1','Toxicity','Cost']

    print("Uploading benchmark graphs ... ",end=':')    
    for i in range(len(metrics_list)):
        print(i,end='|')
        Ben_Res2S3(s3_bucket,task_folder+'_'+metrics_list[i]+'.png',bench_key,task_folder)
    
    # Upload cost-performance plots
    print("\nUploading cost-performance graphs ... ",end=':')  
    Ben_Res2S3(s3_bucket,cp_file_name,bench_key,task_folder)
    Ben_Res2S3(s3_bucket,top_cp_file_name,bench_key,task_folder)
    
    print("\nBenchmark output upload completed.")
    
    return
