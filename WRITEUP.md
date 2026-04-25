name: Keep-alive ping
on:
  schedule:
    - cron: "0 12 * * 1"
  workflow_dispatch:
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Timestamp ping
        run: echo "Rail-Drishti active — $(date -u)" >> docs/PING.log
      - name: Commit ping
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add docs/PING.log
          git diff --staged --quiet || git commit -m "chore: weekly keep-alive ping"
          git push
