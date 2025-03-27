# Task based LLM Evaluation and Migration Framework for enterprise use cases

As Generative AI adoption accelerates, organizations face the challenge of selecting the right model for their specific use cases, given the varied performance across accuracy, coherence, hallucination rates, bias, fairness, latency, and cost. For example, many LLM families including Nova and Claude provide models with different trade-offs between advanced reasoning capabilities and costs versus faster response times and lower costs, with the optimal choice depending on the specific use case. Effective model comparison also requires tailoring prompts for each LLM to ensure fair and accurate evaluations.

Also, we have seen new large language models (LLMs) constantly entering the market, each with unique capabilities, architectures and optimizations. The generative AI practitioners have mission to start migrating off the existing foundation models used in their Generative AI workloads, and adopting the new  models for cost-performance. During the migration, one of the key challenges is ensuring that performance after migration is at least as good as or better than prior to the migration. To achieve this, thorough model evaluation and baselining are essential before migration. Optimizing the prompts on new models are also very important to align performance with that of the previous workload or improve upon them. 

To address these demands, we provide a LLM task based evaluation and migration framework for enterprise use cases, in this repository. 

## Evaluating LLM for model selection

![PeccyBen](/images/Peccyben_framework.png)

* Focus on your evaluation domain: Performance, Responsibility, Infrastructure, Cost 
* Task driven evaluation fitting your use cases: Summarization, Question-Answer by RAG, Classification, Entity Extraction, Sentiment analysis, Open-end question/text generation, Complex reasoning.
* Cover LLMs on AWS: Models on Bedrock, SageMaker, 3P models on HuggingFace, Fine-tuned model vs foundation model
* Insights on model cost-performance based on your selected performance metrics and AWS public pricing
* Help you experiment prompts: multiple prompt tuning and engineering techniques, and test the variation of prompts.
* Integrate with multiple eval tools: leveraging existing evaluation tools, provide holistic views on benchmarking metrics.


## Optimizing prompt for model migration 

Migrating the model from your Generative AI workload to a new LLM requires a structured approach to ensure performance consistency and improvement. It includes evaluating and benchmarking the old and new models, optimizing prompts on the new model, testing and deploying the new models in your production.

![PeccyMig](/images/peccymig_framework.png)

1. Source model evaluation and baselining: evaluate the source model and collect key performance metrics based on your business use case, such as response accuracy, latency, and cost, to set a performance baseline as the model migration target.  
2. Prompt optimization: refine and improve the structure, parameters, and languages of your prompts to adapt to the new Nova model for accurate, relevant and faithful outputs. We will discuss more in the next section. 
3. Target Model Evaluation and continuous optimization: evaluate the optimized prompts on migrated Nova model, to meet the performance target defined in step 1. You can conduct the optimization in step 2 as an iterative process, until the optimized prompts meet your business criteria. 
4. Production testing and deployment: conduct A/B testing to validate the Nova model performance in your testing and production environment. Once satisfied, deploy the Nova model, the settings, and the prompts in production. 

## How to setup and run the library

Please refer to the [runbook here](/RunBook.md)

 
## Example notebooks  

* **Model evaluation** for Summarization task use case: click [here](examples/summarization_example.ipynb).   
* **Model evaluation** for Classification task use case: click [here](examples/classification_example.ipynb).  
* **Model evaluation** for Question-Answer (RAG) task use case: click [here](examples/ragqa_example.ipynb).   
* **Model evaluation** for Sentiment task use case: click [here](examples/sentiment_example.ipynb).  
* **Model migration** for Summarization task use case: click [here](examples/summarizatione_2emig_example.ipynb).   
* **Model migration** for Classification task use case: click [here](examples/classification_e2emig_example.ipynb).  
* **Model migration** for Question-Answer (RAG) task use case: click [here](examples/ragqa_e2emig_example.ipynb).   

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

