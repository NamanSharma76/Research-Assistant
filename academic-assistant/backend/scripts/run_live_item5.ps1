$ErrorActionPreference = "Continue"
$BaseUrl = "http://127.0.0.1:8000"

$state = Get-Content "academic-assistant/backend/data/live_state.json" | ConvertFrom-Json
$tokenB = $state.tokenB
$docIdA = $state.docId
$convIdA = $state.convId

Write-Host "=================================================="
Write-Host "ITEM 5: User B Isolation Tests (Expect 404 on all)"
Write-Host "=================================================="

$endpoints = @(
    @{ Name = "GET /documents/{A_id}"; Method = "GET"; Url = "$BaseUrl/documents/$docIdA"; Body = $null },
    @{ Name = "DELETE /documents/{A_id}"; Method = "DELETE"; Url = "$BaseUrl/documents/$docIdA"; Body = $null },
    @{ Name = "GET /documents/{A_id}/sections"; Method = "GET"; Url = "$BaseUrl/documents/$docIdA/sections"; Body = $null },
    @{ Name = "GET /documents/{A_id}/summary"; Method = "GET"; Url = "$BaseUrl/documents/$docIdA/summary"; Body = $null },
    @{ Name = "GET /documents/{A_id}/notes"; Method = "GET"; Url = "$BaseUrl/documents/$docIdA/notes"; Body = $null },
    @{ Name = "GET /documents/{A_id}/outline"; Method = "GET"; Url = "$BaseUrl/documents/$docIdA/outline"; Body = $null },
    @{ Name = "POST /documents/{A_id}/conversations"; Method = "POST"; Url = "$BaseUrl/documents/$docIdA/conversations"; Body = '{"title":"B stealing A"}' },
    @{ Name = "GET /conversations/{C1}/messages"; Method = "GET"; Url = "$BaseUrl/conversations/$convIdA/messages"; Body = $null },
    @{ Name = "POST /conversations/{C1}/messages"; Method = "POST"; Url = "$BaseUrl/conversations/$convIdA/messages"; Body = '{"content":"B intruder message"}' },
    @{ Name = "DELETE /conversations/{C1}"; Method = "DELETE"; Url = "$BaseUrl/conversations/$convIdA"; Body = $null }
)

foreach ($ep in $endpoints) {
    try {
        $headers = @{
            Authorization = "Bearer $tokenB"
            "Content-Type" = "application/json"
        }
        if ($ep.Method -eq "GET") {
            $resp = Invoke-WebRequest -Uri $ep.Url -Headers $headers -Method Get -SkipHttpErrorCheck
        } elseif ($ep.Method -eq "DELETE") {
            $resp = Invoke-WebRequest -Uri $ep.Url -Headers $headers -Method Delete -SkipHttpErrorCheck
        } elseif ($ep.Method -eq "POST") {
            $resp = Invoke-WebRequest -Uri $ep.Url -Headers $headers -Method Post -Body $ep.Body -SkipHttpErrorCheck
        }
        Write-Host "[$($ep.Name)] -> HTTP_STATUS:$($resp.StatusCode) | Body: $($resp.Content)"
    } catch {
        Write-Host "[$($ep.Name)] -> Error: $_"
    }
}
