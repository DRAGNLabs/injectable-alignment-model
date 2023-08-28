# Rocket Training

Welcome to Rocket Training! In this repository, you'll find all the information you need to set up and train your rocket model using Hugging Face's Llama 2 platform.

### Setup Process

To get started, follow these steps to set up your environment and start training your rocket model:

1. **Create a Hugging Face Account**: If you haven't already, go to [Hugging Face](https://huggingface.co/login) and create an account.

2. **Request Access to Llama 2**: Visit [Llama 2](https://ai.meta.com/llama/) and request access using the same email address as your Hugging Face account. 

3. **Access the Llama-2-7b-hf Model**: Request access to the [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) model through Hugging Face. Your account will be granted access within 0-2 days.

5. **Set Up Your Environment**
    - In the command line, navigate to your preferred directory and clone this repository with the following prompt:
      ```
      git clone https://github.com/DRAGNLabs/Rocket.git
      ```
      
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
      mamba install -c conda-forge --file requirements.txt
      ```
6. **Access Key Setup**
    - Follow these [steps](https://huggingface.co/docs/hub/security-tokens) to set up an read access token.
      
    - In command line, replace your token and run the following command:
      
      ```
      export HF_TOKEN="<YOUR_TOKEN_HERE>"
      ```
7. **Complete Setup**:
    - Run the setup script to finalize the configuration:
      ```
      python setup.py
      ```

    - Submit job.sh to train a test model
      ```
      sbatch job.sh
      ```

You're now all set up to start training your rocket model! If you have any questions or run into issues, feel free to refer to the documentation or reach out to our community for assistance.

Happy rocket training! 🚀🛰️
