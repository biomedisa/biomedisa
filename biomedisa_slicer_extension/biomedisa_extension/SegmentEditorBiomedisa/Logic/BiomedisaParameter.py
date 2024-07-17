from typing import Optional

class BiomedisaParameter():
    nbrw: int = 10
    sorw: int = 4000
    compression: bool = True
    allaxis: bool = False
    denoise: bool = False
    uncertainty: bool = False
    ignore: str = 'none'
    only: str = 'all'
    smooth: int  = 0
    platform: str = None
    return_hits: bool = False
    acwe: bool = False
    acwe_alpha: float = 1.0
    acwe_smooth: int = 1
    acwe_steps: int = 3
    clean: Optional[float] = None
    fill: Optional[float] = None
    
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
        #    f"{indent}smooth: {self.smooth}",
        #    f"{indent}uncertainty: {self.uncertainty}",
        #    f"{indent}compression: {self.compression}",
        #    f"{indent}return_hits: {self.return_hits}",
        #    f"{indent}acwe: {self.acwe}",
        #    f"{indent}acwe_alpha: {self.acwe_alpha}",
        #    f"{indent}acwe_smooth: {self.acwe_smooth}",
        #    f"{indent}acwe_steps: {self.acwe_steps}",
        #    f"{indent}clean: {self.clean}",
        #    f"{indent}fill: {self.fill}",
        ]
        return f"{header}\n" + "\n".join(parameters)
    