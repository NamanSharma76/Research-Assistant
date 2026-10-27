$ErrorActionPreference = "Stop"
$BaseUrl = "http://127.0.0.1:8000"

$state = Get-Content "academic-assistant/backend/data/live_state.json" | ConvertFrom-Json
$tokenA = $state.tokenA
$headersA = @{ Authorization = "Bearer $tokenA" }
$pdfPath = "academic-assistant/backend/data/uploads/attention_is_all_you_need.pdf"

Write-Host "=================================================="
Write-Host "ITEM 8: Concurrent Messages to Same Conversation (200 vs 409)"
Write-Host "=================================================="

# 1. Upload doc for User A
Write-Host "`n--> Uploading document for User A"
$uploadRaw = & curl.exe -s -X POST "$BaseUrl/documents" -H "Authorization: Bearer $tokenA" -F "file=@$pdfPath;type=application/pdf"
$uploadRes = $uploadRaw | ConvertFrom-Json
$docId = $uploadRes.id

for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 1
    $st = Invoke-RestMethod -Uri "$BaseUrl/documents/$docId" -Headers $headersA
    if ($st.status -eq "ready") { break }
}

# 2. Create conversation
$conv = Invoke-RestMethod -Uri "$BaseUrl/documents/$docId/conversations" -Method Post -Headers $headersA -ContentType "application/json" -Body '{"title":"Concurrency Test"}'
$convId = $conv.id
Write-Host "Created Conversation: $convId"

# 3. Launch two simultaneous POST requests via Start-Job
Write-Host "`n--> Launching 2 simultaneous POST /conversations/$convId/messages via Start-Job"

$codeBlock = {
    param($url, $token, $content)
    try {
        $headers = @{
            Authorization = "Bearer $token"
            "Content-Type" = "application/json"
        }
        $body = @{ content = $content } | ConvertTo-Json
        $res = Invoke-RestMethod -Uri $url -Method Post -Headers $headers -Body $body
        return "HTTP 200 OK | Response: " + ($res | ConvertTo-Json -Compress)
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        $stream = $_.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        $respBody = $reader.ReadToEnd()
        return "HTTP $status Conflict | Error Detail: $respBody"
    }
}

$job1 = Start-Job -ScriptBlock $codeBlock -ArgumentList "$BaseUrl/conversations/$convId/messages", $tokenA, "Concurrent turn 1: what is attention?"
$job2 = Start-Job -ScriptBlock $codeBlock -ArgumentList "$BaseUrl/conversations/$convId/messages", $tokenA, "Concurrent turn 2: what is attention?"

Write-Host "Waiting for both background jobs to finish..."
$null = Wait-Job $job1, $job2

$out1 = Receive-Job $job1
$out2 = Receive-Job $job2

Write-Host "`n=== Job 1 Output ==="
Write-Host $out1
Write-Host "`n=== Job 2 Output ==="
Write-Host $out2

Remove-Job $job1, $job2
