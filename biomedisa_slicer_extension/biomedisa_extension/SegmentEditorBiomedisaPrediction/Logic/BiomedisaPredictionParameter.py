class BiomedisaPredictionParameter():
    path_to_model: str
    stride_size: int = 32
    batch_size_active: bool = False
    batch_size: int = 12

    def __str__(self):
        header = "BiomedisaPredictionParameter:"
        indent = "    "  # Four spaces for indentation
        parameters = [
            f"{indent}pathToModel: {self.path_to_model}",
            f"{indent}stride_size: {self.stride_size}",
            f"{indent}batch_size_active: {self.batch_size_active}",
            f"{indent}batch_size: {self.batch_size}", 
        ]
        return f"{header}\n" + "\n".join(parameters)
    
    def to_dict(self):
        """Convert the parameters to a dictionary."""
        return {
            'pathToModel': self.path_to_model,
            'stride_size': self.stride_size,
            'batch_size_active': self.batch_size_active,
            'batch_size': self.batch_size,
        }

    @classmethod
    def from_dict(cls, param_dict):
        """Create an instance from a dictionary."""
        parameter = cls()
        parameter.path_to_model=param_dict.get('pathToModel', "")
        parameter.stride_size=param_dict.get('stride_size', 32)
        parameter.batch_size_active=param_dict.get('batch_size_active', False)
        parameter.batch_size=param_dict.get('batch_size', 12)
        return parameter