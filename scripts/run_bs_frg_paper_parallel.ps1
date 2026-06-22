param(
    [int]$Jobs = 12,
    [double]$TimeLimit = 20,
    [int]$BeamWidth = 1000,
    [int]$K = 3,
    [int]$MaxDepth = 1000,
    [string]$RunName = "bs_frg_pure_paper_full_$(Get-Date -Format yyyyMMdd_HHmmss)"
)

$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $Root

$OutDir = Join-Path $Root 'docs\experiments'
$PartDir = Join-Path $OutDir ($RunName + '_parts')
$Log = Join-Path $OutDir ($RunName + '_run.log')

New-Item -ItemType Directory -Force -Path $PartDir | Out-Null
"START $(Get-Date -Format o) jobs=$Jobs time_limit=$TimeLimit beam_width=$BeamWidth k=$K max_depth=$MaxDepth" |
    Set-Content -LiteralPath $Log -Encoding utf8

$tasks = @()

Get-ChildItem -LiteralPath (Join-Path $Root 'external\paper_cpmp\BF\BF') -Directory |
    Sort-Object Name |
    ForEach-Object {
        $safe = $_.Name -replace '[^A-Za-z0-9_-]', '_'
        $tasks += [pscustomobject]@{
            Dataset = 'BF'
            Family = $_.Name
            Output = Join-Path $PartDir ("BF_${safe}.csv")
            Stdout = Join-Path $PartDir ("BF_${safe}.out.log")
            Stderr = Join-Path $PartDir ("BF_${safe}.err.log")
        }
    }

$cvFamilies = Get-ChildItem -LiteralPath (Join-Path $Root 'external\paper_cpmp\CV\CRPTestcases_Caserta') -Filter 'data*-*-*.dat' |
    ForEach-Object {
        if ($_.BaseName -match '^data(\d+)-(\d+)-\d+$') {
            "$($matches[1])-$($matches[2])"
        }
    } |
    Sort-Object -Unique

$cvFamilies | ForEach-Object {
    $safe = $_ -replace '[^A-Za-z0-9_-]', '_'
    $tasks += [pscustomobject]@{
        Dataset = 'CV'
        Family = $_
        Output = Join-Path $PartDir ("CV_${safe}.csv")
        Stdout = Join-Path $PartDir ("CV_${safe}.out.log")
        Stderr = Join-Path $PartDir ("CV_${safe}.err.log")
    }
}

$bfTaskCount = @($tasks | Where-Object Dataset -eq 'BF').Count
$cvTaskCount = @($tasks | Where-Object Dataset -eq 'CV').Count
"TASKS total=$($tasks.Count) bf=$bfTaskCount cv=$cvTaskCount" | Add-Content -LiteralPath $Log -Encoding utf8

$running = @()
$next = 0
$completed = 0
$failed = @()

while ($completed -lt $tasks.Count) {
    while ($running.Count -lt $Jobs -and $next -lt $tasks.Count) {
        $task = $tasks[$next]
        $next += 1
        $args = @(
            'scripts\benchmark_bs_frg_paper.py',
            '--dataset', $task.Dataset,
            '--families', $task.Family,
            '--time-limit', ([string]$TimeLimit),
            '--beam-width', ([string]$BeamWidth),
            '--k', ([string]$K),
            '--max-depth', ([string]$MaxDepth),
            '--output', $task.Output
        )
        $proc = Start-Process `
            -FilePath 'python' `
            -ArgumentList $args `
            -WorkingDirectory $Root `
            -RedirectStandardOutput $task.Stdout `
            -RedirectStandardError $task.Stderr `
            -WindowStyle Hidden `
            -PassThru
        $running += [pscustomobject]@{ Process = $proc; Task = $task; Start = Get-Date }
        "START_TASK $(Get-Date -Format o) pid=$($proc.Id) $($task.Dataset) $($task.Family)" |
            Add-Content -LiteralPath $Log -Encoding utf8
    }

    Start-Sleep -Seconds 10
    $still = @()
    foreach ($item in $running) {
        $item.Process.Refresh()
        if ($item.Process.HasExited) {
            $item.Process.WaitForExit()
            $completed += 1
            $elapsed = ((Get-Date) - $item.Start).TotalSeconds
            $exitCode = $item.Process.ExitCode
            "DONE_TASK $(Get-Date -Format o) exit=$exitCode elapsed=$([Math]::Round($elapsed, 1)) completed=$completed/$($tasks.Count) $($item.Task.Dataset) $($item.Task.Family)" |
                Add-Content -LiteralPath $Log -Encoding utf8
            if ($exitCode -ne 0) {
                $failed += $item.Task
            }
        } else {
            $still += $item
        }
    }
    $running = $still
    "HEARTBEAT $(Get-Date -Format o) running=$($running.Count) queued=$($tasks.Count - $next) completed=$completed failed=$($failed.Count)" |
        Add-Content -LiteralPath $Log -Encoding utf8
}

$combined = Join-Path $OutDir ($RunName + '.csv')
$files = Get-ChildItem -LiteralPath $PartDir -Filter '*.csv' | Sort-Object Name
$first = $true
if (Test-Path -LiteralPath $combined) {
    Remove-Item -LiteralPath $combined -Force
}
foreach ($file in $files) {
    $lines = Get-Content -LiteralPath $file.FullName
    if ($lines.Count -eq 0) {
        continue
    }
    if ($first) {
        $lines | Set-Content -LiteralPath $combined -Encoding utf8
        $first = $false
    } else {
        $lines | Select-Object -Skip 1 | Add-Content -LiteralPath $combined -Encoding utf8
    }
}

$rows = @(Import-Csv -LiteralPath $combined)
$solved = @($rows | Where-Object { $_.goal_validation_ok -eq 'True' -and $_.final_n -eq '0' -and $_.moves -ne '' })
$unsolved = @($rows | Where-Object { !(($_.goal_validation_ok -eq 'True') -and ($_.final_n -eq '0') -and ($_.moves -ne '')) })

$moveSum = 0
foreach ($row in $solved) {
    $moveSum += [int]$row.moves
}
$runtimeSum = 0.0
foreach ($row in $rows) {
    $runtimeSum += [double]$row.runtime_sec
}
$avgRuntime = if ($rows.Count -gt 0) { $runtimeSum / $rows.Count } else { 0.0 }

$summary = Join-Path $OutDir ($RunName + '_summary.md')
$summaryLines = @(
    '# Paper BS-FRG Full Benchmark Summary',
    '',
    "Result CSV: ``$combined``",
    '',
    '| solver | solved | moves_sum | avg_runtime_sec | failed_family_jobs |',
    '|---|---:|---:|---:|---:|',
    "| bs_frg_pure | $($solved.Count)/$($rows.Count) | $moveSum | $([Math]::Round($avgRuntime, 3)) | $($failed.Count) |",
    '',
    '## Unsolved',
    ''
)
if ($unsolved.Count -eq 0) {
    $summaryLines += '- none'
} else {
    foreach ($row in ($unsolved | Select-Object -First 200)) {
        $summaryLines += "- ``$($row.family)/$($row.case)``: $($row.status) final_n=$($row.final_n) moves=$($row.moves)"
    }
    if ($unsolved.Count -gt 200) {
        $summaryLines += "- ... $($unsolved.Count - 200) more"
    }
}
$summaryLines | Set-Content -LiteralPath $summary -Encoding utf8

"COMBINED rows=$($rows.Count) solved=$($solved.Count) moves_sum=$moveSum avg_runtime=$([Math]::Round($avgRuntime, 3)) failed_jobs=$($failed.Count)" |
    Add-Content -LiteralPath $Log -Encoding utf8

if ($failed.Count -gt 0) {
    $failedLabels = ($failed | ForEach-Object { "$($_.Dataset):$($_.Family)" }) -join ','
    "FAILED_FAMILIES $failedLabels" | Add-Content -LiteralPath $Log -Encoding utf8
    exit 1
}

"FINISH $(Get-Date -Format o)" | Add-Content -LiteralPath $Log -Encoding utf8
