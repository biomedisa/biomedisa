#### Set Path Variable using PowerShell
Open PowerShell as administrator (e.g. Windows Search `PowerShell`) and run:
```
$currentPath = [System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine)
$newPath = Resolve-Path -Path "C:\Program Files\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\Hostx64\x64" | select -ExpandProperty Path
$newPathValue = "$currentPath;$newPath"
[System.Environment]::SetEnvironmentVariable('PATH', $newPathValue, [System.EnvironmentVariableTarget]::Machine)
```
