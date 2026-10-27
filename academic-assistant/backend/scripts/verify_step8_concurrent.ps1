$loginBody = @{
    email = "live_user1@example.com"
    password = "SuperSecretPassword123"
} | ConvertTo-Json

$loginRes = Invoke-RestMethod -Uri "http://127.0.0.1:8000/auth/login" -Method Post -Body $loginBody -ContentType "application/json"
$token = $loginRes.access_token
$headers = @{ Authorization = "Bearer $token" }

$convs = Invoke-RestMethod -Uri "http://127.0.0.1:8000/conversations" -Headers $headers
$convId = $convs[0].id

Write-Host "Triggering two simultaneous POST /conversations/$convId/messages jobs..."

$sb = {
    param($token, $convId, $msg)
    $headers = @{ Authorization = "Bearer $token" }
    $body = @{ content = $msg } | ConvertTo-Json
    try {
        $res = Invoke-RestMethod -Uri "http://127.0.0.1:8000/conversations/$convId/messages" -Method Post -Headers $headers -Body $body -ContentType "application/json"
        return "SUCCESS: 200 - $($res.intent)"
    } catch {
        return "CONFLICT: $($_.Exception.Response.StatusCode.value__) - $($_.ErrorDetails.Message)"
    }
}

$job1 = Start-Job -ScriptBlock $sb -ArgumentList $token, $convId, "Job message 1"
$job2 = Start-Job -ScriptBlock $sb -ArgumentList $token, $convId, "Job message 2"

$res1 = Receive-Job -Job $job1 -Wait
$res2 = Receive-Job -Job $job2 -Wait

Write-Host "Job 1 result: $res1"
Write-Host "Job 2 result: $res2"

Remove-Job -Job $job1, $job2
