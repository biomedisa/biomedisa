class BiomedisaPredictionParameter():
    path_to_model: str = ""
    stride_size: int = 32
    batch_size_active: bool = False
    batch_size: int = 12
    x_min: int = None
    x_max: int = None
    y_min: int = None
    y_max: int = None
    z_min: int = None
    z_max: int = None

    def __str__(self):
        header = "BiomedisaPredictionParameter:"
        indent = "    "  # Four spaces for indentation
        parameters = [
            f"{indent}path_to_model: {self.path_to_model}",
            f"{indent}stride_size: {self.stride_size}",
            f"{indent}batch_size_active: {self.batch_size_active}",
            f"{indent}batch_size: {self.batch_size}", 
            f"{indent}x_min: {self.x_min}",
            f"{indent}x_max: {self.x_max}",
            f"{indent}y_min: {self.y_min}",
            f"{indent}y_max: {self.y_max}",
            f"{indent}z_min: {self.z_min}",
            f"{indent}z_max: {self.z_max}"
        ]
        return f"{header}\n" + "\n".join(parameters)
    
    def to_dict(self):
        """Convert the parameters to a dictionary."""
        return {
            'path_to_model': self.path_to_model,
            'stride_size': self.stride_size,
            'batch_size_active': self.batch_size_active,
            'batch_size': self.batch_size,
            'x_min': self.x_min,
            'x_max': self.x_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'z_min': self.z_min,
            'z_max': self.z_max
        }

    @classmethod
    def from_dict(cls, param_dict):
        """Create an instance from a dictionary."""
        parameter = cls()
        parameter.path_to_model=param_dict.get('path_to_model', "")
        parameter.stride_size=param_dict.get('stride_size', 32)
        parameter.batch_size_active=param_dict.get('batch_size_active', False)
        parameter.batch_size=param_dict.get('batch_size', 12)
        parameter.x_min = param_dict.get('x_min', None)
        parameter.x_max = param_dict.get('x_max', None)
        parameter.y_min = param_dict.get('y_min', None)
        parameter.y_max = param_dict.get('y_max', None)
        parameter.z_min = param_dict.get('z_min', None)
        parameter.z_max = param_dict.get('z_max', None)
        return parameter