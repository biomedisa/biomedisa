class BiomedisaTrainingParameter:
    def __init__(self):
        self.path_to_model: str = ""
        self.stride_size: int = 32
        self.epochs: int = 100
        self.validation_split: float = 0.0
        self.balance: bool = False
        self.swapaxes: bool = False
        self.flip_x: bool = False
        self.flip_y: bool = False
        self.flip_z: bool = False
        self.scaling: bool = False
        self.x_scale: int = 256
        self.y_scale: int = 256
        self.z_scale: int = 256

    def __str__(self):
        header = "BiomedisaTrainingParameter:"
        indent = "    "  # Four spaces for indentation
        parameters = [
            f"{indent}path_to_model: {self.path_to_model}",
            f"{indent}stride_size: {self.stride_size}",
            f"{indent}epochs: {self.epochs}",
            f"{indent}validation_split: {self.validation_split}",
            f"{indent}balance: {self.balance}",
            f"{indent}swapaxes: {self.swapaxes}",
            f"{indent}flip_x: {self.flip_x}",
            f"{indent}flip_y: {self.flip_y}",
            f"{indent}flip_z: {self.flip_z}",
            f"{indent}scaling: {self.scaling}",
            f"{indent}x_scale: {self.x_scale}",
            f"{indent}y_scale: {self.y_scale}",
            f"{indent}z_scale: {self.z_scale}",
        ]
        return f"{header}\n" + "\n".join(parameters)
    
    def to_dict(self):
        """Convert the parameters to a dictionary."""
        return {
            'path_to_model': self.path_to_model,
            'stride_size': self.stride_size,
            'epochs': self.epochs,
            'validation_split': self.validation_split,
            'balance': self.balance,
            'swapaxes': self.swapaxes,
            'flip_x': self.flip_x,
            'flip_y': self.flip_y,
            'flip_z': self.flip_z,
            'scaling': self.scaling,
            'x_scale': self.x_scale,
            'y_scale': self.y_scale,
            'z_scale': self.z_scale,
        }

    @classmethod
    def from_dict(cls, param_dict):
        """Create an instance from a dictionary."""
        parameter = cls()
        parameter.path_to_model = param_dict.get('path_to_model', "")
        parameter.stride_size = param_dict.get('stride_size', 32)
        parameter.epochs = param_dict.get('epochs', 100)
        parameter.validation_split = param_dict.get('validation_split', 0.0)
        parameter.balance = param_dict.get('balance', False)
        parameter.swapaxes = param_dict.get('swapaxes', False)
        parameter.flip_x = param_dict.get('flip_x', False)
        parameter.flip_y = param_dict.get('flip_y', False)
        parameter.flip_z = param_dict.get('flip_z', False)
        parameter.scaling = param_dict.get('scaling', False)
        parameter.x_scale = param_dict.get('x_scale', 256)
        parameter.y_scale = param_dict.get('y_scale', 256)
        parameter.z_scale = param_dict.get('z_scale', 256)
        return parameter