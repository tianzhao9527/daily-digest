# Daily Digest (静态预生成 + GitHub Pages)

这个仓库会在 **北京时间每天 08:00** 自动生成当天的「一页简报」，并部署到 GitHub Pages。
你打开网页时只是阅读静态 HTML，不再在浏览器端抓取，因此不会被 CORS/反爬影响。

## 你需要做的事（一次性）

1. 在 GitHub 新建一个仓库（建议名：`daily-digest`）。
2. 把本仓库内容上传/推送到该仓库的 `main` 分支。
3. 进入仓库 **Settings → Pages**
   - **Build and deployment → Source** 选择 **GitHub Actions**
4. 等待 Actions 运行完成（或手动触发一次）：
   - Actions → `Daily Digest (Beijing 08:00)` → Run workflow

完成后，你会得到固定网址：
- `https://<你的GitHub用户名>.github.io/<仓库名>/`

## 页面说明

- `index.html`：永远指向“最新一次生成”的版本（固定入口，建议收藏）
- `daily_digest_YYYY-MM-DD.html`：按天归档的版本

## 手动生成（本地可选）

```bash
python3 scripts/news_digest.py --publish-dir public
open public/index.html
```

## 自定义（可选）

- 你可以在 `scripts/news_digest.py` 里调整栏目、RSS 源、筛选权重、Top10 逻辑等。
- 如果要绑定自定义域名：
  1) Settings → Pages → Custom domain  
  2) 在 `public/` 里生成一个 `CNAME` 文件（内容为你的域名），并在生成逻辑里保留它。
