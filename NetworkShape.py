class NetworkShape:

    # number of neurons in the input layer
    k: int
    input_layer_activation_function: str
    hidden_layers_shape: dict
    """
    example:
    
    {
        1: {
            'n': 10,
            'activation_function': 'sigmoid'
        },

        2: {
            'n': 10,
            'activation_function': 'sigmoid'
        },
        
        # ...
    }
    """
    J: int
    output_layer_activation_function: str

    def __int__(self, k: int, J: int,
                input_layer_activation_function: str,
                hidden_layers_shape: dict,
                output_layer_activation_function: str):

        self.k = k
        self.J = J
        self.input_layer_activation_function = input_layer_activation_function
        self.hidden_layers_shape = hidden_layers_shape
        self.output_layer_activation_function = output_layer_activation_function

