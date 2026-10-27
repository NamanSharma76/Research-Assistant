$ErrorActionPreference = "Stop"

Write-Host "=========================================="
Write-Host "1. Register + Login User 1"
Write-Host "=========================================="
$regBody = @{
    email = "live_user1@example.com"
    password = "SuperSecretPassword123"
} | ConvertTo-Json

try {
    $regRes = Invoke-RestMethod -Uri "http://127.0.0.1:8000/auth/register" -Method Post -Body $regBody -ContentType "application/json"
    Write-Host "Registered: $($regRes | ConvertTo-Json -Compress)"
} catch {
    Write-Host "User already registered or error: $_"
}

$loginBody = @{
    email = "live_user1@example.com"
    password = "SuperSecretPassword123"
} | ConvertTo-Json

$loginRes = Invoke-RestMethod -Uri "http://127.0.0.1:8000/auth/login" -Method Post -Body $loginBody -ContentType "application/json"
$token1 = $loginRes.access_token
$truncatedToken = $token1.Substring(0, [Math]::Min(25, $token1.Length)) + "..."
Write-Host "Token (truncated): $truncatedToken"

$headers1 = @{
    Authorization = "Bearer $token1"
}

Write-Host "`n=========================================="
Write-Host "2. Upload attention_is_all_you_need.pdf"
Write-Host "=========================================="
$pdfPath = "academic-assistant\backend\data\sample_papers\attention_is_all_you_need.pdf"

$uploadRes = curl.exe -s -X POST "http://127.0.0.1:8000/documents" `
  -H "Authorization: Bearer $token1" `
  -F "file=@$pdfPath;type=application/pdf" | ConvertFrom-Json

$docId = $uploadRes.id
Write-Host "Upload response: id=$docId status=$($uploadRes.status) doc_id=$($uploadRes.doc_id)"

Write-Host "Polling GET /documents/$docId until ready..."
for ($i = 0; $i -lt 30; $i++) {
    $poll = Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents/$docId" -Headers $headers1
    Write-Host "Status: $($poll.status) | Chunks: $($poll.num_chunks)"
    if ($poll.status -eq "ready" -or $poll.status -eq "failed") {
        break
    }
    Start-Sleep -Seconds 2
}

Write-Host "`n=========================================="
Write-Host "3. Create Conversation & Send 3 Messages"
Write-Host "=========================================="
$convBody = @{ title = "Attention Paper Chat" } | ConvertTo-Json
$conv = Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents/$docId/conversations" -Method Post -Headers $headers1 -Body $convBody -ContentType "application/json"
$convId = $conv.id
Write-Host "Created Conversation ID: $convId"

$questions = @(
    "What dataset did they use for translation?",
    "explain that more simply",
    "what is the capital of France?"
)

foreach ($q in $questions) {
    Write-Host "`n--- Sending: $q ---"
    $msgBody = @{ content = $q } | ConvertTo-Json
    $msgRes = Invoke-RestMethod -Uri "http://127.0.0.1:8000/conversations/$convId/messages" -Method Post -Headers $headers1 -Body $msgBody -ContentType "application/json"
    Write-Host "Intent: $($msgRes.intent)"
    Write-Host "Trace: $($msgRes.trace -join ' -> ')"
    Write-Host "LLM Calls: $($msgRes.llm_calls) | Time: $($msgRes.seconds)s"
    $ansPreview = $msgRes.content
    if ($ansPreview.Length -gt 250) {
        $ansPreview = $ansPreview.Substring(0, 250) + "..."
    }
    Write-Host "Answer: $ansPreview"
}

Write-Host "`n=========================================="
Write-Host "4. GET notes and GET outline?slides=6"
Write-Host "=========================================="
$notesRes = Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents/$docId/notes" -Headers $headers1
Write-Host "Notes Problem preview: $($notesRes.problem.text.Substring(0, [Math]::Min(150, $notesRes.problem.text.Length)))..."

$outlineRes = Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents/$docId/outline?slides=6" -Headers $headers1
Write-Host "Outline Title: $($outlineRes.title)"
Write-Host "Slide Count: $($outlineRes.slides.Count)"
foreach ($s in $outlineRes.slides) {
    Write-Host "  Slide $($s.slide_number): $($s.title) [Source: $($s.source)]"
}

Write-Host "`n=========================================="
Write-Host "5. Register User 2 & Ownership Check (404)"
Write-Host "=========================================="
$reg2Body = @{
    email = "live_user2@example.com"
    password = "DifferentUserPassword456"
} | ConvertTo-Json
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/auth/register" -Method Post -Body $reg2Body -ContentType "application/json" | Out-Null
} catch {}

$login2Res = Invoke-RestMethod -Uri "http://127.0.0.1:8000/auth/login" -Method Post -Body $reg2Body -ContentType "application/json"
$token2 = $login2Res.access_token
$headers2 = @{ Authorization = "Bearer $token2" }

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/documents/$docId" -Headers $headers2
    Write-Host "ERROR: User 2 accessed User 1's document!"
} catch {
    Write-Host "User 2 GET User 1's document response code: $($_.Exception.Response.StatusCode.value__)"
}

Write-Host "`n=========================================="
Write-Host "6. Delete Document & Check Chunks Removed"
Write-Host "=========================================="
$delRes = curl.exe -s -w "%{http_code}" -X DELETE "http://127.0.0.1:8000/documents/$docId" -H "Authorization: Bearer $token1"
Write-Host "Delete response HTTP status: $delRes"

$afterDel = curl.exe -s -w "%{http_code}" -X GET "http://127.0.0.1:8000/documents/$docId" -H "Authorization: Bearer $token1"
Write-Host "GET after delete HTTP status: $afterDel"
