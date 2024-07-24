from symbol import parameters
from typing import Optional

class BiomedisaParameter():
    allaxis: bool = False
    denoise: bool = False
    nbrw: int = 10
    sorw: int = 4000
    ignore: str = 'none'
    only: str = 'all'
    platform: str = None
  
    def __str__(self):
        header = "BiomedisaParameter:"
        indent = "    "  # Four spaces for indentation
        parameters = [
            f"{indent}allaxis: {self.allaxis}",
            f"{indent}denoise: {self.denoise}",
            f"{indent}nbrw: {self.nbrw}",
            f"{indent}sorw: {self.sorw}", 
            f"{indent}ignore: {self.ignore}",
            f"{indent}only: {self.only}",
            f"{indent}platform: {self.platform}",
        ]
        return f"{header}\n" + "\n".join(parameters)
    
    def to_dict(self):
        """Convert the parameters to a dictionary."""
        return {
            'allaxis': self.allaxis,
            'denoise': self.denoise,
            'nbrw': self.nbrw,
            'sorw': self.sorw,
            'ignore': self.ignore,
            'only': self.only,
            'platform': self.platform
        }

    @classmethod
    def from_dict(self, param_dict):
        """Create an instance from a dictionary."""
        parameter = self()
        parameter.allaxis=param_dict.get('allaxis', False)
        parameter.denoise=param_dict.get('denoise', False)
        parameter.nbrw=param_dict.get('nbrw', 10)
        parameter.sorw=param_dict.get('sorw', 4000)
        parameter.ignore=param_dict.get('ignore', 'none')
        parameter.only=param_dict.get('only', 'all')
        parameter.platform=param_dict.get('platform', None)
        return parameter