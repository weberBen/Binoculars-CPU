# **CPU Inference of Binoculars (Zero-Shot Detection of LLM-Generated Text)** [[demo]](https://huggingface.co/spaces/ben-weber/Binoculars-CPU)

This project adapts the [Binoculars](https://github.com/ahans30/Binoculars) code, which it heavily rely on, to run efficiently on CPUs by leveraging smaller language models for both the observer and reference models. Specifically, it uses the [`SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) language model.

See the online demo : [here 🚀](https://huggingface.co/spaces/ben-weber/Binoculars-CPU) and go to the app url (you might need to wait for the app to restart after a long period of nonactivity). Note that, short-length content may require similar processing time as long-length content because only the latter benefits from parallelization.

![Demo Interface](assets/gradio-interface.png)

![Demo API](assets/api_docs_interface.png)

---

The app is a text analysis tool that handles both raw text and PDFs through a GUI or developer-friendly API. Process single documents or batch analyze multiples files with secure token authentication. Optimized for CPU with GPU support available for enhanced performance.

Get started in minutes with no code required for installation with HuggingFace Space !

See the original paper : [Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text](https://arxiv.org/abs/2401.12070).

---

## **Performance**

- **Accuracy**: On the `datasets` benchmark, we achieve **85% accuracy** with `SmolLM2-135M`.
- **Throughput**:
  - **HuggingFace free CPU Basic (2 vCPU x 16GB RAM)**: ~170 tokens/second (5.9 seconds for 1,000 tokens).
  - **Consumer-grade CPU 4x2.60GHz (16GB RAM)**: ~10 tokens/second (1 minute 30 seconds for 1,000 tokens).

In fact, this code does not requires 16GB of RAM but consumes around 3GB.

---

## **Usage**

 - Navigate to the HuggingFace url or in a local install case to `http://127.0.0.1:7860` in your web browser to access the app (defined in `app.py`).
 - See the GUI interface at `/app`
 - See the API doc at `/docs`.
   - Default api key is `my_api_key_1`.
   - You can test the API with `client.py` which allow you to run Binoculars either on raw text or with a pdf.

### **API usage**

See API docs at `/docs` for the real route, arguments and response usage.

**Request:**

   ```python
   data = {
      "contents": [
         "my_text_1",
         "my_text_2"
      ]
   }

   requests.post(f"{API_URL}/predict", data=data)
   ```

OR

   ```python
   files = [
      ('files', open('file1.pdf', 'rb')),
      ('files', open('file2.pdf', 'rb'))
   ]

   requests.post(f"{API_URL}/predict", files=files)
   ```

**Response:**

   ```json
   {
      "total_gpu_time": 12.35552716255188,
      "total_token_count": 2081,
      "model": {
         "threshold": 0.99963529763794,
         "observer_model": "HuggingFaceTB/SmolLM2-135M",
         "performer_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
      },
      "results": [
         {
            "score": 0.8846334218978882,
            "class_label": 0,
            "label": "Most likely AI-generated",
            "content_length": 661,
            "chunk_count": 1
         },
         {
            "..."
         }
      ]
   }
   ```

### **Model usage**

```python
from binoculars import Binoculars

bino = Binoculars()

pred_class, pred_label, score = bino.predict(
   "my text", return_fields=["class", "label", "score"])

print(pred_class, pred_label, score) # Most likely AI-generated, 0, 0.8846334218978882
```

---

## **Installation**

Keep in mind that the same instance of Binoculars model is being used by all requests (GUI/API) defined in `interface/bino_singleton.py`. In fact instantiating 2 instances of Binoculars model will requires 2x more VRAM or CPU usage/Memory.

### **HuggingFace deployment**

#### **No code**

You can use the no code install by cloning the HuggingFace Space [here](https://huggingface.co/spaces/ben-weber/Binoculars-CPU).

#### **Manual**

You can manually install the HuggingFace Space :

- Create a new HuggingFace Space
- Select the desired hardware (the application will run on the `CPU Basic` free hardware tier).
- Clone this repository to your HuggingFace Space.
- Rename this `README.md` to `README-doc.md`.
- Rename `README-HuggingFace-Space.md` to `README.md`.
- Set the env variable (see the production mode section).
- Run the factory rebuild.

#### **Notes**

- You can switch between hardware on HuggingFace from CPU to GPU without impacting the inner working of this code.
- If you want to run the application on a private HuggingFace space you can enable the dev mode and make a ssh port forwarding :
```bash
ssh -L 7860:127.0.0.1:7860 username@ssh.hf.space
```
And then go to `127.0.0.1:7860`.

### **Local setup**

1. **Build and run the Docker container**:
   ```bash
   docker compose build
   docker compose up
   ```

And then go to `127.0.0.1:7860`.

You can also run model directly with :

```bash
docker compose exec binoculars bash -c "python main.py"
```

Note that you can enforce CPU usage even if GPU/cuda is available on your computer by setting the `BINOCULARS_FORCE_TO_CPU` environment variable.

---

## **Production mode**

You need to change the following env variables:

- `API_AUTHORIZED_KEYS` : authorized key for auth (equivalent of password)
- `API_SECRET_KEY` : secret key used for the API token encryption
- `API_ACCESS_TOKEN_EXPIRE_MINUTES` : duration of the bearer token
- `HF_HOME` : HuggingFace transformers cache directory where models are download (make sure to set it to persistent storage in order to avoid re-download after each startup. Only mandatory when using large model, not small model like the default one of this repo.)
- `MODEL_CHUNK_SIZE` : chunk size in char to feed the model with
- `MODEL_BATCH_SIZE` : number of chunks that can be processed as a batch.
- `BINOCULARS_OBSERVER_MODEL_NAME` : HuggingFace model name of the observer
- `BINOCULARS_PERFORMER_MODEL_NAME` : HuggingFace model name of performer
- `BINOCULARS_THRESHOLD` : Binoculars detection threshold between AI/Human content
- `MAX_FILE_SIZE_BYTES` : maximum file size allowed
- `MODEL_MINIMUM_TOKENS` : minimum allowed number of tokens per document. Short text show a low accuracy rate
- `FLATTEN_BATCH` : allow document chunks to be flatten across all document for optimized batch processing. May introduce slightly score variation compare to individual processing. See usage notes section.


See all the available variables in `config.py`.

Notes 
1. Note that HuggingFace cold boot [can takes serval minutes](https://discuss.huggingface.co/t/slow-space-cold-boot/72154), which need to be taken into account for the hardware pricing and sleep time (after the initial startup that need to download models).
2. Note that the total input data moved to VRAM GPU is `MODEL_CHUNK_SIZE * MODEL_BATCH_SIZE`.
3. Changing models (observer/performer) involves adapting the threshold to theses specific models. See section on model customization.

---

## **Model customization**

To change the models:

1. **Observer and Reference models**:
   You can change the models (observer/performer) with environment variables `BINOCULARS_OBSERVER_MODEL_NAME` and `BINOCULARS_PERFORMER_MODEL_NAME`. Be aware that the same tokenizer is used for both the model (which are base model and instruct model).

2. **Update the threshold**:
   Adjust the environment variable `BINOCULARS_THRESHOLD`. This threshold is tied to the specific models used and affects performance:
   - You can use the simple `threshold_finder.py` script to calculate an optimal threshold. This script analyzes the original Binoculars datasets, minimizing mismatches between target class and prediction using MSE loss with a sigmoid function for soft (differentiable) ranking.
   - The `threshold_finder.py` script requires `.csv` files generated by `experiments/jobs.sh`.
   - For simplicity, the code employs a single threshold for high accuracy. However, the original Binoculars paper recommends using two thresholds for optimizing either low false positive rates or high accuracy.

Models used in the original paper were `tiiuae/falcon-7b` and `tiiuae/falcon-7b-instruct` with threshold optimize for accuracy `0.9015310749276843` or optimize for low false positive rate `0.8536432310785527`.

---

## **Notes**

- Short-length content may require similar processing time as long-length content because only the latter benefits from parallelization.
- When initializing models, you may encounter the following warning:
  ```
  Some weights of LlamaForCausalLM were not initialized from the model checkpoint at HuggingFaceTB/SmolLM2-135M and are newly initialized.
  ```
  This message is safe to **ignore**. It does not impact the model's runtime or accuracy ([see here](https://huggingface.co/LeoLM/leo-hessianai-13b-chat/discussions/3), [and there](https://huggingface.co/codellama/CodeLlama-7b-hf/discussions/1)).
- When server is started you will see a "spamming" process in the uvicorn log that ping the route home `/`. It's the `init-proc` process of HuggingFace that start the uvicorn process. **It does not affect the sleep timeout of the HuggingFace space. But do not keep an active session open (where user can interact with or the focus tab of a browser)**.
- For a document that exceeds the chunk limit, it will be cut into multiple chunks. The associated document score is the average of all chunk scores.
- When using batch processing, sequences within a batch are padded to match the length of the longest sequence in order to create a rectangular tensor, which hardware is optimized to process. Consequently, you may observe slight numerical differences (on the order of 1e-3) in results when comparing batch processing to individual inference.
- When running the same model on different hardware architectures, you may observe score differences (on the order of 1e-3) due to hardware-specific architectures and their implementations of floating-point arithmetic.
