@echo off
REM Batch script to compile all LaTeX reports
REM Binary Speech Emotion Recognition Project
REM Authors: Ammar Qurthuby & Habibi

echo ========================================
echo  COMPILING ALL PROJECT REPORTS
echo ========================================
echo.

cd /d "d:\SEMESTER 7 & S2 1\Skripsi\Klasifikasi Suara\IEEE_MTT_S_Conference__IMS_2013_example___1_"

echo [1/3] Compiling Proposal Report...
echo -------------------------------------
pdflatex -interaction=nonstopmode proposal_report.tex
bibtex proposal_report
pdflatex -interaction=nonstopmode proposal_report.tex
pdflatex -interaction=nonstopmode proposal_report.tex
echo.

echo [2/3] Compiling Progress Report...
echo -------------------------------------
pdflatex -interaction=nonstopmode progress_report.tex
bibtex progress_report
pdflatex -interaction=nonstopmode progress_report.tex
pdflatex -interaction=nonstopmode progress_report.tex
echo.

echo [3/3] Compiling Final Report...
echo -------------------------------------
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo.

echo ========================================
echo  COMPILATION COMPLETE
echo ========================================
echo.
echo Generated PDFs:
echo  - proposal_report.pdf
echo  - progress_report.pdf
echo  - main.pdf
echo.

pause
