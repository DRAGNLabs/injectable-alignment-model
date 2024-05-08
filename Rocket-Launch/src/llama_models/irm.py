import torch.nn as nn
import torch
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move up one level
parent_dir = os.path.dirname(current_dir)

# Construct the path to the module
module_path = os.path.join(parent_dir, "utils", "tensor_logger.py")

# Import the module dynamically (advanced technique)
import importlib.util
spec = importlib.util.spec_from_file_location("tensor_logger", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# from transformers import LlamaConfig
#from tensor_logger import tensor_logger


class IRM(nn.Module):
    def __init__(self, config, size_modifier = 3):
        super(IRM, self).__init__()
        self.weights: torch.Tensor = []
        self.device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')
        self.logger = module.tensor_logger(config.model_config["hidden_size"])

        self.vocab_size = config.vocab_size
        self.hidden_size = config.model_config["hidden_size"]
        self.linear_size = self.hidden_size * size_modifier


        # self.batch_size = config.batch_size
        self.sequence_length = config.model_config["max_position_embeddings"]

        self.injection_layers = config.IRM_layers
        self.num_layers = len(self.injection_layers)
        self.active_irm = True

        self.basic_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.linear_size),
            nn.ReLU(),
            nn.Linear(self.linear_size, self.linear_size),
            nn.ReLU(),
            nn.Linear(self.linear_size, self.linear_size),
            nn.ReLU(),
            nn.Linear(self.linear_size, self.linear_size),
            nn.ReLU(),
            nn.Linear(self.linear_size, self.hidden_size * self.num_layers),
        ).to(self.device)
        
    def forward(self, x: torch.Tensor):
        curr_batch_size = x.size()[0]
        self.weights = self.basic_forward(x).view(curr_batch_size, -1, self.hidden_size, self.num_layers)
        # print(self.weights.size())
        # self.logger.add_tensor(self.weights)

    def get_layer_weights(self, layer_id):
        return self.weights[:, :, :, self.injection_layers.index(layer_id)]

    def activate(self):
        self.active_irm = True

    def deactivate(self):
        self.active_irm = False

    def injected_operation(self, layer_id, llm_output):
        if self.active_irm:
            return self.get_layer_weights(layer_id) + llm_output
        else:
            return llm_output

    def logModel(self):
        self.logger.new_prompt()
        
        self.logger.write_log()
        self.logger.generate_heatmap()
        # self.logger.generate_histograms()
        # self.logger.hard_coded_graph()


if __name__ == "__main__":
    # model = IRM(LlamaConfig())
    # # model.forward(torch.randn((1,1024,512)))
    #model.forward(torch.randn((1,1024,512)))
    # print(model.weights[3])
    # model = IRM(LlamaConfig(vocab_size=30522, max_position_embeddings=512, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12))
    # test_input = torch.randn((1, 1024, 768)).to(model.device)
    # test_input2 = torch.randn((1, 1024, 768)).to(model.device)
    # test_input3 = torch.randn((1, 1024, 768)).to(model.device)
    print("howdy")
    # model.forward(test_input)
    # model.forward(test_input2)
    # model.forward(test_input3)

    # model.logModel()