# Rocket Training

Welcome to Rocket Training! In this repository, you'll find all the information you need to set up and train your rocket model using Hugging Face's Llama 2 platform.

### Setup Process

To get started, follow these steps to set up your environment and start training your rocket model:

1. **Create a Hugging Face Account**: If you haven't already, go to [Hugging Face](https://huggingface.co/login) and create an account.

2. **Request Access to Llama 2**: Visit [Llama 2](https://ai.meta.com/llama/) and request access using the same email address as your Hugging Face account. 

3. **Access the Llama-2-7b-chat-hf Model**: Request access to the 'Llama-2-7b-chat-hf' model through Hugging Face. Your account will be granted access to all available versions within 1-2 days.

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
    This script will tokenize the dataset, initiate a brief training sequence, and save a model to the "Models" folder.

You're now all set up to start training your rocket model! If you have any questions or run into issues, feel free to refer to the documentation or reach out to our community for assistance.

Happy rocket training! 🚀🛰️