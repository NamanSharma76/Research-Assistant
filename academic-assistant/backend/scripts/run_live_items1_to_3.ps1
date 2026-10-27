$ErrorActionPreference = "Stop"
$BaseUrl = "http://127.0.0.1:8000"

Write-Host "=================================================="
Write-Host "ITEM 1: Register User A and User B, and Login Both"
Write-Host "=================================================="

# 1. Register User A
Write-Host "`n--> POST /auth/register (User A: usera@test.com)"
$resA_reg = Invoke-RestMethod -Uri "$BaseUrl/auth/register" -Method Post -ContentType "application/json" -Body '{"email":"usera@test.com","password":"PasswordA_12345!"}'
$resA_reg | ConvertTo-Json -Depth 5

# 2. Register User B
Write-Host "`n--> POST /auth/register (User B: userb@test.com)"
$resB_reg = Invoke-RestMethod -Uri "$BaseUrl/auth/register" -Method Post -ContentType "application/json" -Body '{"email":"userb@test.com","password":"PasswordB_12345!"}'
$resB_reg | ConvertTo-Json -Depth 5

# 3. Login User A
Write-Host "`n--> POST /auth/login (User A)"
$resA_login = Invoke-RestMethod -Uri "$BaseUrl/auth/login" -Method Post -ContentType "application/json" -Body '{"email":"usera@test.com","password":"PasswordA_12345!"}'
$tokenA = $resA_login.access_token
Write-Host "User A Token received: $($tokenA.Substring(0, 30))... (Length: $($tokenA.Length))"

# 4. Login User B
Write-Host "`n--> POST /auth/login (User B)"
$resB_login = Invoke-RestMethod -Uri "$BaseUrl/auth/login" -Method Post -ContentType "application/json" -Body '{"email":"userb@test.com","password":"PasswordB_12345!"}'
$tokenB = $resB_login.access_token
Write-Host "User B Token received: $($tokenB.Substring(0, 30))... (Length: $($tokenB.Length))"

$headersA = @{ Authorization = "Bearer $tokenA" }
$headersB = @{ Authorization = "Bearer $tokenB" }

Write-Host "`n=================================================="
Write-Host "ITEM 2: User A Uploads PDF & Polls until Ready"
Write-Host "=================================================="

$pdfPath = "academic-assistant/backend/data/uploads/attention_is_all_you_need.pdf"
Write-Host "`n--> Uploading $pdfPath as User A via curl.exe"
$uploadResRaw = & curl.exe -s -X POST "$BaseUrl/documents" -H "Authorization: Bearer $tokenA" -F "file=@$pdfPath;type=application/pdf"
$uploadRes = $uploadResRaw | ConvertFrom-Json
$uploadRes | ConvertTo-Json -Depth 5

$docId = $uploadRes.id
Write-Host "Document DB ID: $docId"

# Poll until ready
Write-Host "`n--> Polling GET /documents/$docId until status == ready"
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    $docStatus = Invoke-RestMethod -Uri "$BaseUrl/documents/$docId" -Headers $headersA
    Write-Host "Poll [$i]: status=$($docStatus.status), num_chunks=$($docStatus.num_chunks), error=$($docStatus.error)"
    if ($docStatus.status -eq "ready" -or $docStatus.status -eq "failed") {
        break
    }
}

Write-Host "`nFinal Document Status Output:"
$docStatus | ConvertTo-Json -Depth 5

Write-Host "`n=================================================="
Write-Host "ITEM 3: User A Creates C1 & Sends 5 Turn Messages"
Write-Host "=================================================="

Write-Host "`n--> POST /documents/$docId/conversations (Title: 'Attention Paper Chat')"
$convRes = Invoke-RestMethod -Uri "$BaseUrl/documents/$docId/conversations" -Method Post -Headers $headersA -ContentType "application/json" -Body '{"title":"Attention Paper Chat"}'
$convRes | ConvertTo-Json -Depth 5
$convId = $convRes.id
Write-Host "Conversation ID: $convId"

$messages = @(
    "What dataset did they use for translation?",
    "explain that more simply",
    "summarize section 6",
    "shorter",
    "what is the capital of France?"
)

foreach ($msg in $messages) {
    Write-Host "`n--------------------------------------------------"
    Write-Host "User message: `"$msg`""
    Write-Host "--------------------------------------------------"
    Start-Sleep -Seconds 2
    $body = @{ content = $msg } | ConvertTo-Json
    $t0 = [System.Diagnostics.Stopwatch]::StartNew()
    $resMsg = Invoke-RestMethod -Uri "$BaseUrl/conversations/$convId/messages" -Method Post -Headers $headersA -ContentType "application/json" -Body $body
    $t0.Stop()
    
    Write-Host "Raw Response:"
    $resMsg | ConvertTo-Json -Depth 5
    Write-Host "`nTurn Summary: Intent: $($resMsg.intent) | Kind: $($resMsg.kind) | LLM Calls: $($resMsg.llm_calls) | Time: $($resMsg.seconds) s | Trace: $($resMsg.trace -join ' -> ')"
}

# Save IDs for item 4-8
@{
    tokenA = $tokenA
    tokenB = $tokenB
    docId = $docId
    convId = $convId
} | ConvertTo-Json | Set-Content -Path "academic-assistant/backend/data/live_state.json"
