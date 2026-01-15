# Test embedding API
$body = @{ content = "Hello world" } | ConvertTo-Json
$response = Invoke-RestMethod -Uri "http://EWRSPT-AI:8083/embedding" -Method POST -ContentType "application/json" -Body $body

Write-Host "Response type: $($response.GetType().Name)"
if ($response -is [array]) {
    Write-Host "Array length: $($response.Count)"
    Write-Host "First element has embedding: $($response[0].embedding -ne $null)"
    if ($response[0].embedding) {
        if ($response[0].embedding -is [array] -and $response[0].embedding[0] -is [array]) {
            Write-Host "Embedding dimensions: $($response[0].embedding[0].Count)"
        } else {
            Write-Host "Embedding dimensions: $($response[0].embedding.Count)"
        }
    }
} else {
    Write-Host "Has embedding: $($response.embedding -ne $null)"
    if ($response.embedding) {
        Write-Host "Embedding dimensions: $($response.embedding.Count)"
    }
}
