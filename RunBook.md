# Runbook

We provide you a runbook to help you start using this library for your own use case and dataset.  

## Install the library

You can use the python environment in Amazon SageMaker Studio, SageMaker Notebook Instance, or setup your own python environment in an Amazon EC2 instance, or your local compute environment with AWS client. 
1. You need to enable AWS IAM permissions for Bedrock, refer to the instruction [here](https://github.com/aws-samples/amazon-bedrock-workshop/tree/main?tab=readme-ov-file#enable-aws-iam-permissions-for-bedrock).   
2. Also, make sure you install the latest version of boto3 library. Use `pip install -U boto3` to upgrade if needed.  

When you have setup the python environment, `git clone` this repo to you local directory. Now you have the following directory structure 
- examples: the example notebooks, you can create your new notebooks under this directory, or create a parallel work directories for your own notebooks. Under each work directory, you need to have the following two subdirectories   
   - data: store the dataset for benchmarking, downloaded from S3 bucket 
   - results: store the benchmarking output data, to be uploaded to S3 bucket. This directory contains task sub-dirs, you can create new sub-dirs for your new use cases, please make sure the sub-dir names are the same name as the sub-folders in S3 bucket created in the next step.   
- peccyben: python library for model evaluation
- peccymig: python library for model migration

## Install the requested public python libraries 

You can open a new notebook under the work directory (or examples directory), and run the following command 
```
!pip install -r ../peccyben/requirements.txt 
```

## Prepare a S3 bucket to store the input and output data.  

Create a new S3 bucket to store the input and output data. Under the bucket, create one folder for each use case/task. For example 

![Alt text](/images/S3_bucket_folders.png)

In each use case folder, you need create two subfolders: the `input-data` folder stores the input data, the `output-results` folder stores multiple testing result sets in sub-folders. 

![Alt text](/images/s3_subfolders.png)

## Upload your use case dataset to S3

Follow [this guide](/data/README.md) to prepare your use case specific dataset, and upload your data to the `input-data` sub-folder in the specific use case folder.  

## Prepare your prompt catalog 

You can manage your prompts in the [prompt catalog](/examples/prompt_catalog.json). In the catalog, you can define each prompt in four sections 
* persona: define role based prompting in this section, to assign a persona to the LLM to guide the style, tone and focus of the response. 
* instruction: define your instructions for the use case in this section, be precise and avoid vague or ambiguous language, tell the LLM exactly what you want it to do  
* inputs: define inputs/contexts placeholders in this section, and desired output format. 
* example: you can add one-shot or few-shot examples in this section. 

## Cost file 

For the cost-performance analysis, you can provide a cost file in a csv file in the `examples` directory or your own work directory. The data format 
- model: the Bedrock model ID
- input_token: the cost/price per 1000000 input tokens
- output_token: the cost/price per 1000000 input tokens

You can refer to for the [Bedrock model public pricing](https://aws.amazon.com/bedrock/pricing/). 

## Set model access on Amazon Bedrock 

For the models you want to evaluate, as well as the models you select for LLM-judge, refer to [this instruction](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access-modify.html) to configure the model access on Amazon Bedrock. 
- You need to get access to the judge models from Amazon Bedrock on AWS region us-west-2. 
- You can configure access to the models selected for benchmarking from Amazon Bedrock on the AWS region you configured in your notebooks. 

## Run the model evaluation and model migration 

Now you can create your own notebook or python script in your environment. You can do it in the `examples` folder, or create your own work directory. If you use SageMaker Studio or Notebook Instance, we recommend you to select     
- image = PyTorch 2.0.0 Python 3.10 GPU Optimized.   
- Kernel = Python 3.
- Instance type = ml.g4dn.xlarge (or other instance type you prefer).

Please refer to the example notebooks 
* **Model evaluation** for Summarization task use case: click [here](examples/summarization_example.ipynb).   
* **Model evaluation** for Classification task use case: click [here](examples/classification_example.ipynb).  
* **Model evaluation** for Question-Answer (RAG) task use case: click [here](examples/ragqa_example.ipynb).   
* **Model evaluation** for Sentiment task use case: click [here](examples/sentiment_example.ipynb).  

## Collect your model evaluation results from S3

The model benchmarking results are stored in the `output-results` folder. Under `output-results`, you can have multiple sub-folders for multiple test rounds.  

![Alt text](/images/s3_output_subfolders.png)

The results contain 3 types of files 

![Alt text](/images/s3_output_files.png)

- Benchmarking metrics across multiple models selected 
- Graphs for each use case specific metric 
- Model response data files
 
## Run the model migration and collect the optimized prompt for the target model 

Please refer to the example notebooks 
* **Model migration** for Summarization task use case: click [here](examples/summarizatione_2emig_example.ipynb).   
* **Model migration** for Classification task use case: click [here](examples/classification_e2emig_example.ipynb).  
* **Model migration** for Question-Answer (RAG) task use case: click [here](examples/ragqa_e2emig_example.ipynb).

You can get the optimized prompt from the output of the function `peccymig.data_aware_optimization`.

## Developer's guide 

If you are interested in developing new use cases, or want to learn more details of the peccyben and peccymig libraries, please refer to the Developer's guide [here](DevGuide.md).  







