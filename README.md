# TritonCodeDataset

This dataset contains extracted **Triton GPU kernels** from various HuggingFace transformer models, compiled using **PyTorch Inductor**.

| **HuggingFace Model** | **Generated Triton File** |
|------------------------|---------------------------|
| HuggingFace/GPTNeoForSequenceClassification | `triton_gptneo.py` |
| HuggingFace/LayoutLMForMaskedLM | `triton_layoutlm_mlm.py` |
| HuggingFace/LayoutLMForSequenceClassification | `triton_layoutlm_seq.py` |
| HuggingFace/M2M100ForConditionalGeneration | `triton_m2m100.py` |
| HuggingFace/MBartForCausalLM | `triton_mbart_clm.py` |
| HuggingFace/MBartForConditionalGeneration | `triton_mbart_cond.py` |
| HuggingFace/MegatronBertForCausalLM | `triton_megatronbert_clm.py` |
| HuggingFace/MegatronBertForQuestionAnswering | `triton_megatronbert_qa.py` |
| HuggingFace/MobileBertForMaskedLM | `triton_mobilebert_mlm.py` |
| HuggingFace/MobileBertForQuestionAnswering | `triton_mobilebert_qa.py` |
| HuggingFace/OPTForCausalLM | `triton_opt.py` |
| HuggingFace/PegasusForCausalLM | `triton_pegasus_clm.py` |
| HuggingFace/PegasusForConditionalGeneration | `triton_pegasus_cond.py` |
| HuggingFace/PLBartForCausalLM | `triton_plbart_clm.py` |
| HuggingFace/PLBartForConditionalGeneration | `triton_plbart_cond.py` |
| HuggingFace/RobertaForCausalLM | `triton_roberta_clm.py` |
| HuggingFace/RobertaForQuestionAnswering | `triton_roberta_qa.py` |
| HuggingFace/XGLMForCausalLM | `triton_xglm.py` |
| HuggingFace/XLNetLMHeadModel | `triton_xlnet.py` |
| HuggingFace/YituTechConvBert | `triton_yitutechconvbert.py` |
