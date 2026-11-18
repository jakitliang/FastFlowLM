param(
    [Parameter(Mandatory=$true)][string]$ModelHome,
    [ValidateRange(1,65535)][int]$Port = 52626,
    [string]$BindAddress = "127.0.0.1",
    [string]$AllowedOrigins = "",
    [string]$XrtRuntimeDir = ""
)
$ErrorActionPreference = "Stop"
$homePath = (Resolve-Path -LiteralPath $ModelHome).Path
if (-not (Test-Path -LiteralPath (Join-Path $homePath "models") -PathType Container)) {
    throw "ModelHome must contain the existing models directory (normally the .flm directory)."
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use; the existing server will not be stopped."
}
$root = Split-Path -Parent $PSScriptRoot
$names = @("PATH", "FLM_MODEL_PATH", "FLM_CONFIG_PATH", "FLM_MODELINFO_PATH", "FLM_XCLBIN_PATH", "FLM_NO_AUTO_PULL")
$saved = @{}
foreach ($name in $names) { $saved[$name] = [Environment]::GetEnvironmentVariable($name, "Process") }
try {
    if ($XrtRuntimeDir) {
        $runtime = (Resolve-Path -LiteralPath $XrtRuntimeDir).Path
        if (-not (Test-Path -LiteralPath (Join-Path $runtime "xrt_coreutil.dll"))) { throw "XRT runtime DLL not found." }
        $env:PATH = "$runtime;$env:PATH"
    }
    $env:FLM_MODEL_PATH = $homePath
    $env:FLM_CONFIG_PATH = Join-Path $root "model_list.json"
    $env:FLM_MODELINFO_PATH = Join-Path $root "model_info.json"
    $env:FLM_XCLBIN_PATH = $root
    $env:FLM_NO_AUTO_PULL = "1"
    $arguments = @("serve", "--host", $BindAddress, "--port", "$Port", "--backend", "flm")
    if ($AllowedOrigins) { $arguments += @("--cors-origins", $AllowedOrigins) }
    Write-Host "Isolated test server on ${BindAddress}:$Port; automatic model pulls disabled. Ctrl+C stops it."
    & (Join-Path $root "flm.exe") @arguments
    if ($LASTEXITCODE -ne 0) { throw "FLM exited with code $LASTEXITCODE" }
} finally {
    foreach ($name in $names) { [Environment]::SetEnvironmentVariable($name, $saved[$name], "Process") }
}