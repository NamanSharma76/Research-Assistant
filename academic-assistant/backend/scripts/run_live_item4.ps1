$ErrorActionPreference = "Stop"
$BaseUrl = "http://127.0.0.1:8000"

$state = Get-Content "academic-assistant/backend/data/live_state.json" | ConvertFrom-Json
$tokenA = $state.tokenA
$convId = $state.convId
$headersA = @{ Authorization = "Bearer $tokenA" }

Write-Host "=================================================="
Write-Host "ITEM 4: Verify Persistence Across Server Restart"
Write-Host "=================================================="

Write-Host "`n--> GET /conversations/$convId/messages"
$msgs = Invoke-RestMethod -Uri "$BaseUrl/conversations/$convId/messages" -Headers $headersA
Write-Host "Total messages retrieved after restart: $($msgs.Count)"
$msgs | ConvertTo-Json -Depth 5

Write-Host "`n--> Sending post-restart message: 'what about the limitations?'"
$body = @{ content = "what about the limitations?" } | ConvertTo-Json
$resMsg = Invoke-RestMethod -Uri "$BaseUrl/conversations/$convId/messages" -Method Post -Headers $headersA -ContentType "application/json" -Body $body

Write-Host "`nPost-restart message response:"
$resMsg | ConvertTo-Json -Depth 5
Write-Host "`nTurn Summary: Intent: $($resMsg.intent) | Kind: $($resMsg.kind) | LLM Calls: $($resMsg.llm_calls) | Time: $($resMsg.seconds) s | Trace: $($resMsg.trace -join ' -> ')"
