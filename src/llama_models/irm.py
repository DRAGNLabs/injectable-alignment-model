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

        # For manual injection
        self.manually_inject = False
        self.injected_neuron_ids : list = []
        self.injected_neuron_vals : list = [] # Must either have the same length as self.injected_neuron_ids or a single value (broadcast)
        self.manual_injection_tensor : torch.Tensor = None

        # Store regularization configuration
        self.regularization_parameters = getattr(config, 'regularize_parameters', False)
        self.regularization_outputs = getattr(config, 'regularize_outputs', False)
        self.regularization_type = getattr(config, 'regularization_type', 'none')  # 'none', 'l1', or 'l2'
        self.regularization_strength = getattr(config, 'regularization_strength', 1e-3)

        # Store injected module configuration
        self.injected_module = getattr(config, 'injected_module', 'mlp')


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
    
    # L2 regularization is more common to be implemented thru the "weight_decay" in optimizer.
    # L2 in optimizer is also implemented in injected_llama_for_causal.py
    def get_regularization_loss(self):
        """
        Calculate regularization loss based on configured type and strength.
        Applies regularization to the parameters of the linear layers, not the outputs.
        Returns 0 if regularization is disabled.
        """
        if self.regularization_type == 'none' or self.num_injected_positions == 0:
            return 0.0
        
        
        # Get all parameters from the basic_forward module
        reg_loss = 0.0

        if self.regularization_parameters:
            for param in self.basic_forward.parameters():
                if self.regularization_type == 'l1':
                    # L1 regularization: sum of absolute values
                    reg_loss += torch.sum(torch.abs(param))
                elif self.regularization_type == 'l2':
                    # L2 regularization: sum of squared values
                    reg_loss += torch.sum(param ** 2) / 2

        if self.regularization_outputs:
            if self.regularization_type == 'l1':
                reg_loss += torch.sum(torch.abs(self.weights))
            elif self.regularization_type == 'l2':
                reg_loss += torch.sum(self.weights ** 2) / 2
        
        return self.regularization_strength * reg_loss

    # FIXME: Right now we are assuming only injecting in one layer
    def setup_manual_injection(self, input_shape):
        if not self.manually_inject:
            return
        
        output_dim = self.hidden_size * self.num_injected_positions
        self.manual_injection_tensor = torch.zeros(input_shape)

        # Check if we need to broadcast values or use them directly
        if len(self.injected_neuron_vals) == 1 and len(self.injected_neuron_ids) > 1:
            # Broadcasting case: one value for all specified neurons
            broadcast_value = self.injected_neuron_vals[0]
            
            # For each batch and sequence position, set the specified neurons to their values
            for neuron_idx in self.injected_neuron_ids:
                # This will set the value at all batch items and sequence positions
                # but only at the specified neuron indices in the hidden dimension
                self.manual_injection_tensor[..., neuron_idx] = broadcast_value
        else:
            # Direct mapping case: each neuron gets its corresponding value
            assert len(self.injected_neuron_ids) == len(self.injected_neuron_vals), \
                "injected_neuron_ids and injected_neuron_vals must have the same length"
            
            # Set each neuron to its specified value
            for idx, (neuron_idx, neuron_val) in enumerate(zip(self.injected_neuron_ids, self.injected_neuron_vals)):
                # Set the value for all batch items and sequence positions
                self.manual_injection_tensor[..., neuron_idx] = neuron_val

        return 

    def forward(self, x: torch.Tensor):
        curr_batch_size = x.size()[0]
        if not self.manually_inject:
            self.weights = self.basic_forward(x).view(curr_batch_size, -1, self.hidden_size, self.num_injected_positions)

            if self.do_logging:
                print("Tensor shape: ", self.weights.size())
                self.logger.add_tensor(self.weights)
                
                # Weights.size() tells you how many layers you have.
                
                # The final dimension is the layers, so you can index the weights by layer.  self.weights[:,:,:,:0] would give you the weights for the first layer.
        else:
            hidden_states_shape = x.shape
            self.setup_manual_injection(hidden_states_shape)
            self.weights = self.manual_injection_tensor.view(curr_batch_size, -1, self.hidden_size, self.num_injected_positions)


    def get_layer_weights(self, layer_id):
        return self.weights[:, :, :, self.injection_layers.index(layer_id)]

    def activate(self):
        self.active_irm = True

    def deactivate(self):
        self.active_irm = False

    def injected_operation(self, layer_id, llm_output): # FIXME: Figure out the shape of llm_output to perform the injection!
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