# PowerShell script to compile all LaTeX reports
# Binary Speech Emotion Recognition Project
# Authors: Ammar Qurthuby & Habibi

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " COMPILING ALL PROJECT REPORTS" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$projectDir = "d:\SEMESTER 7 & S2 1\Skripsi\Klasifikasi Suara\IEEE_MTT_S_Conference__IMS_2013_example___1_"
Set-Location $projectDir

function Compile-Report {
    param (
        [string]$ReportName,
        [int]$Number
    )
    
    Write-Host "[$Number/3] Compiling $ReportName..." -ForegroundColor Yellow
    Write-Host "-------------------------------------" -ForegroundColor Gray
    
    # First pass
    pdflatex -interaction=nonstopmode "$ReportName.tex" | Out-Null
    
    # BibTeX
    bibtex $ReportName | Out-Null
    
    # Second pass
    pdflatex -interaction=nonstopmode "$ReportName.tex" | Out-Null
    
    # Third pass
    pdflatex -interaction=nonstopmode "$ReportName.tex" | Out-Null
    
    if (Test-Path "$ReportName.pdf") {
        $fileSize = (Get-Item "$ReportName.pdf").Length
        $fileSizeKB = [math]::Round($fileSize / 1KB, 2)
        Write-Host "✓ Success! Generated $ReportName.pdf ($fileSizeKB KB)" -ForegroundColor Green
    } else {
        Write-Host "✗ Error: $ReportName.pdf not generated" -ForegroundColor Red
    }
    Write-Host ""
}

# Compile all reports
Compile-Report "proposal_report" 1
Compile-Report "progress_report" 2
Compile-Report "main" 3

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " COMPILATION COMPLETE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Generated PDFs:" -ForegroundColor White
if (Test-Path "proposal_report.pdf") {
    $size1 = [math]::Round((Get-Item "proposal_report.pdf").Length / 1KB, 2)
    Write-Host "  ✓ proposal_report.pdf ($size1 KB)" -ForegroundColor Green
}
if (Test-Path "progress_report.pdf") {
    $size2 = [math]::Round((Get-Item "progress_report.pdf").Length / 1KB, 2)
    Write-Host "  ✓ progress_report.pdf ($size2 KB)" -ForegroundColor Green
}
if (Test-Path "main.pdf") {
    $size3 = [math]::Round((Get-Item "main.pdf").Length / 1KB, 2)
    Write-Host "  ✓ main.pdf ($size3 KB)" -ForegroundColor Green
}
Write-Host ""

Read-Host "Press Enter to exit"
