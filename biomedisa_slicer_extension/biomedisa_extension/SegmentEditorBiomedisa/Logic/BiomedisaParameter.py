class BiomedisaParameter():
    allaxis: bool = False
    denoise: bool = False
    nbrw: int = 10
    sorw: int = 4000
    smooth_active: bool = False
    smooth: int = 100
    clean_active: bool = False
    clean: float = 0.1
    fill_active: bool = False
    fill: float = 0.9
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
            f"{indent}smooth_active: {self.smooth_active}",
            f"{indent}smooth: {self.smooth}",
            f"{indent}clean_active: {self.clean_active}",
            f"{indent}clean: {self.clean}",
            f"{indent}fill_active: {self.fill_active}",
            f"{indent}fill: {self.fill}",
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
            'smooth_active': self.smooth_active,
            'smooth': self.smooth,
            'clean_active': self.clean_active,
            'clean': self.clean,
            'fill_active': self.fill_active,
            'fill': self.fill,
            'ignore': self.ignore,
            'only': self.only,
            'platform': self.platform
        }

    @classmethod
    def from_dict(cls, param_dict):
        """Create an instance from a dictionary."""
        parameter = cls()
        parameter.allaxis=param_dict.get('allaxis', False)
        parameter.denoise=param_dict.get('denoise', False)
        parameter.nbrw=param_dict.get('nbrw', 10)
        parameter.sorw=param_dict.get('sorw', 4000)
        parameter.smooth_active=param_dict.get('smooth_active', False)
        parameter.smooth=param_dict.get('smooth', 100)
        parameter.clean_active=param_dict.get('clean_active', False)
        parameter.clean=param_dict.get('clean', 0.1)
        parameter.fill_active=param_dict.get('fill_active', False)
        parameter.fill=param_dict.get('fill', 0.9)
        parameter.ignore=param_dict.get('ignore', 'none')
        parameter.only=param_dict.get('only', 'all')
        parameter.platform=param_dict.get('platform', None)
        return parameter

