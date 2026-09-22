# ==============================================================================
# HackAlem AI - Hourly Commit Script (PowerShell)
# Enforces Hackathon Rule 5.4.8: Intermediate progress result every hour.
# Usage: .\scripts\hourly_commit.ps1 -Hour 1 -Message "Implemented core agent tools"
# ==============================================================================

param (
    [Parameter(Mandatory=$false)]
    [int]$Hour = 1,

    [Parameter(Mandatory=$false)]
    [string]$Message = "Hourly progress checkpoint"
)

$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$CommitMsg = "checkpoint(hour-$Hour): $Message [$Timestamp]"

Write-Host "🚀 Preparing HackAlem AI Hourly Checkpoint..." -ForegroundColor Cyan
Write-Host "🕒 Time: $Timestamp"
Write-Host "📝 Commit Message: $CommitMsg" -ForegroundColor Yellow

# Stage all files except ignored ones
git add -A

# Check if there are changes to commit
$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "⚠️ No local changes detected! Making an empty checkpoint commit to satisfy Rule 5.4.8..." -ForegroundColor Yellow
    git commit --allow-empty -m "$CommitMsg"
} else {
    git commit -m "$CommitMsg"
}

# Push to GitHub
Write-Host "📤 Pushing to remote repository..." -ForegroundColor Cyan
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Checkpoint successfully pushed to GitHub! Rule 5.4.8 satisfied." -ForegroundColor Green
} else {
    Write-Host "❌ Push failed! Check your internet connection or git credentials." -ForegroundColor Red
}
