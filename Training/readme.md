# Rocket Training

Welcome to Rocket Training! In this repository, you'll find all the information you need to set up and train your rocket model using Hugging Face's Llama 2 platform.

### Setup Process

1. **Create a Hugging Face Account**: If you haven't already, go to [Hugging Face](https://huggingface.co/login) and create an account.

2. **Request Access to Llama 2**: Visit [Llama 2](https://ai.meta.com/llama/) and request access using the same email address as your Hugging Face account. 

3. **Access the Llama-2-7b-hf Model**: Request access to the [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model through Hugging Face. Your account will be granted access within 1-2 days.

4. **Set Up Your Environment**:
    - Run the following command to create a new environment with Python 3.11:
      ```
      mamba create --name rocket_training python=3.11
      ```
    - Activate the newly created environment:
      ```
      mamba activate rocket_training
      ```
    - Install the required dependencies from the provided `requirements.txt` file:
      ```
      pip install -r requirements.txt
      ```

5. **Complete Setup**:
    - Run the setup script to finalize the configuration:
      ```
      python setup.py
      ```

    - Submit job.sh to train a test model
      ```
      sbatch job.sh
      ```

You're now all set up to start training your rocket model!

Happy rocket training! 🚀🛰️