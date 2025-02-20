import torch.nn as nn
import torch
import os


from utils.tensor_logger import tensor_logger


# Get the current script's directory
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Move up one level
# parent_dir = os.path.dirname(current_dir)

# # Construct the path to the module
# module_path = os.path.join(parent_dir, "utils", "tensor_logger.py")

# # Import the module dynamically (advanced technique)
# import importlib.util
# spec = importlib.util.spec_from_file_location("tensor_logger", module_path)
# module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(module)

# from transformers import LlamaConfig
#from tensor_logger import tensor_logger


class IRM(nn.Module):
    def __init__(self, config, size_modifier = 3):
        super(IRM, self).__init__()
        self.weights: torch.Tensor = []
        self.device = torch.device('cuda:0' if 'CUDA_VISIBLE_DEVICES' in os.environ else 'cpu')

        self.do_logging = config.do_logging


        self.vocab_size = config.vocab_size
        self.hidden_size = config.model_config["hidden_size"]
        self.num_linear_layers = config.model_config["irm_layer_num"]
        if config.model_config["irm_layer_size"] == -1:
            self.linear_size = self.hidden_size * size_modifier
        else:
            self.linear_size = config.model_config["irm_layer_size"]

        # self.batch_size = config.batch_size
        self.sequence_length = config.model_config["max_position_embeddings"]

        self.injection_layers = config.IRM_layers
        self.num_injected_positions = len(self.injection_layers)
        self.active_irm = True

        if len(self.injection_layers) == 0:
            self.deactivate()

        if self.do_logging:
            self.logger = tensor_logger(config.model_config["num_hidden_layers"], config.experiment_name, self.injection_layers, config.default_root_dir)
            # Pass self.num_injected_positions and self.injection_layers to the tensor_logger constructor
        else:
            self.logger = None
            
        self.basic_forward = self.create_sequential_model().to(self.device)

    def create_sequential_model(self):
        if self.num_injected_positions == 0:
            return nn.Sequential()
        
        layers = []
        
        if self.num_linear_layers == 1:
            # If only one layer, directly connect input to output
            layers.append(nn.Linear(self.hidden_size, self.hidden_size * self.num_injected_positions))
        else:
            # First layer
            layers.append(nn.Linear(self.hidden_size, self.linear_size))
            layers.append(nn.ReLU())
            
            # Middle layers
            for _ in range(self.num_linear_layers - 2):
                layers.append(nn.Linear(self.linear_size, self.linear_size))
                layers.append(nn.ReLU())
            
            # Last layer
            layers.append(nn.Linear(self.linear_size, self.hidden_size * self.num_injected_positions))
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        curr_batch_size = x.size()[0]
        self.weights = self.basic_forward(x).view(curr_batch_size, -1, self.hidden_size, self.num_injected_positions)

        if self.do_logging:
            print("Tensor shape: ", self.weights.size())
            self.logger.add_tensor(self.weights)
            
			# Weights.size() tells you how many layers you have.
            
			# The final dimension is the layers, so you can index the weights by layer.  self.weights[:,:,:,:0] would give you the weights for the first layer.


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
        if (self.do_logging):
            self.logger.new_prompt()
            # self.logger.write_log()
            self.logger.generate_heatmaps()
            # self.logger.generate_histograms()
        
    def logSparsityPlot(self):
        if (self.do_logging): self.logger.sparcity_graph_per_token()



if __name__ == "__main__":
    # model = IRM(LlamaConfig())
    # # model.forward(torch.randn((1,1024,512)))
    #model.forward(torch.randn((1,1024,512)))
    # print(model.weights[3])
    # model = IRM(LlamaConfig(vocab_size=30522, max_position_embeddings=512, hidden_size=768, intermediate_size=3072, num_hidden_layers=32, num_attention_heads=12))

    # test_input = torch.randn((1, 1024, 768)).to(model.device)
    # test_input2 = torch.randn((1, 1024, 768)).to(model.device)
    # test_input3 = torch.randn((1, 1024, 768)).to(model.device)
    print("howdy")
    # model.forward(test_input)
    # model.forward(test_input2)
    # model.forward(test_input3)

    # model.logModel()