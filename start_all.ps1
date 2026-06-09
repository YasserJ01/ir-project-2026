# start_all.ps1 — Launch all 7 IR project services
# Run this from the repo root:  .\start_all.ps1
# Each service opens its own PowerShell window with visible logs.
# Gateawy: http://localhost:8000/docs
# UI:      http://localhost:5173

$root = "F:\IR project"
$venvActivate = "$root\.venv\Scripts\Activate.ps1"
$uiDir = "$root\services\ui"

$backendServices = @(
    @{name="preprocessing"; port=8001; mod="services.preprocessing.app.pipeline:app"}
    @{name="indexing";       port=8002; mod="services.indexing.app.service:app"}
    @{name="retrieval";      port=8003; mod="services.retrieval.app.service:app"}
    @{name="refinement";     port=8004; mod="services.refinement.app.service:app"}
    @{name="rag";            port=8005; mod="services.rag.app.main:app"}
    @{name="gateway";        port=8000; mod="services.gateway.app.main:app"}
)

Write-Host "=== IR Project — Starting All Services ===" -ForegroundColor Cyan
Write-Host ""

# 1. Start backend services in dependency order
foreach ($svc in $backendServices) {
    $title = "IR: $($svc.name) (:$(svc.port))"
    Write-Host "  Starting $title ..." -ForegroundColor Yellow
    $arg = "-NoExit", "-Command", "& '$venvActivate'; uvicorn $($svc.mod) --host 0.0.0.0 --port $($svc.port)"
    Start-Process powershell -WindowStyle Normal -ArgumentList $arg
    Start-Sleep -Seconds 3
}

# 2. Start React UI (npm run dev)
Write-Host "  Starting UI (:5173) ..." -ForegroundColor Yellow
$uiArg = "-NoExit", "-Command", "cd '$uiDir'; npm run dev"
Start-Process powershell -WindowStyle Normal -ArgumentList $uiArg

Write-Host ""
Write-Host "=== All services launched ===" -ForegroundColor Green
Write-Host "  Gateway: http://localhost:8000/docs"
Write-Host "  UI:      http://localhost:5173"
