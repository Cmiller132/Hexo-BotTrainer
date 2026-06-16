# Elevated: forward host 0.0.0.0:8080 -> WSL dashboard, + inbound firewall rule.
$ErrorActionPreference = 'Continue'
$wslip = '172.26.60.178'
$result = 'C:\Users\epicm\portproxy_result.txt'
netsh interface portproxy delete v4tov4 listenport=8080 listenaddress=0.0.0.0 2>$null | Out-Null
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$wslip | Out-Null
Remove-NetFirewallRule -DisplayName 'WSL Hexo Dashboard 8080' -ErrorAction SilentlyContinue
New-NetFirewallRule -DisplayName 'WSL Hexo Dashboard 8080' -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow | Out-Null
$out = netsh interface portproxy show v4tov4 | Out-String
Set-Content -Path $result -Value $out -Encoding utf8
Add-Content -Path $result -Value 'PORTPROXY_DONE'
