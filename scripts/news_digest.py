#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_digest_v5.py

V5：新增“汇率 / 利率 / 大宗（金属）”趋势面板（sparkline）
- 默认监控：USDCNY、10YUSY.B、DX.F、HG.F、X8.F(铝EU溢价，作为铝的代理)
- 统一从 Stooq CSV 拉取（无需Key）；若某个品种拉取失败，会自动跳过但不影响页面生成
- 10Y收益率的“变化”用 bp（基点）呈现更符合直觉

运行：
  python3 news_digest_v5.py --out daily_digest.html
  python3 news_digest_v5.py --demo --out daily_digest.html
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import difflib
import gzip
import io
import json
import math
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------
# 栏目与来源配置
# -----------------------------

SECTIONS_ORDER: List[str] = [
    "Top 10 今日要点",
    "宏观与政策（全球）",
    "地缘政治与制裁合规",
    "AI 基础设施与算力链",
    "企业软件与搜索/RAG",
    "跨境供应链与大宗/再生金属",
    "企业采购与福利",
    "金融市场与风险",
    "消费品牌与零售",
    "材料与可持续",
    "碳市场与ESG",
    "硬件与IoT",
    "人才与组织",
    "观点与言论（高信噪比）",
    "机会雷达（Deal/合作/并购/融资）",
    "异常与预警（触发才显示）",
    "论文雷达：机器学习（cs.LG）",
    "论文雷达：经济学（econ.GN）",
    "论文雷达：计算机与社会（cs.CY）",
]

RSS_FEEDS: Dict[str, List[Tuple[str, str]]] = {
    "宏观与政策（全球）": [
        ("IMF", "https://www.imf.org/external/rss/feeds.aspx?feed=imfnews"),
        ("World Bank", "https://www.worldbank.org/en/news/all?format=rss"),
        ("OECD", "https://www.oecd.org/newsroom/rss.xml"),
    ],
    "地缘政治与制裁合规": [
        ("UN News", "https://news.un.org/feed/subscribe/en/news/all/rss.xml"),
        ("US Treasury", "https://home.treasury.gov/feed"),
        ("EU Council", "https://www.consilium.europa.eu/en/press/press-releases/rss/"),
    ],
    "AI 基础设施与算力链": [
        ("TechCrunch", "https://techcrunch.com/feed/"),
        ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ],
    "企业软件与搜索/RAG": [
        ("InfoQ", "https://www.infoq.com/feed/"),
    ],
    "跨境供应链与大宗/再生金属": [
        ("Lloyd's List (public)", "https://www.lloydslist.com/LL113545.rss"),
    ],
    "企业采购与福利": [
        ("Finextra", "https://www.finextra.com/rss/headlines.aspx"),
    ],
    "金融市场与风险": [
        ("Investopedia", "https://www.investopedia.com/rss/news.rss"),
    ],
    "消费品牌与零售": [
        ("Retail Dive", "https://www.retaildive.com/feeds/news/"),
    ],
    "材料与可持续": [
        ("GreenBiz", "https://www.greenbiz.com/rss.xml"),
    ],
    "碳市场与ESG": [
        ("Carbon Brief", "https://www.carbonbrief.org/feed/"),
    ],
    "硬件与IoT": [
        ("IEEE Spectrum", "https://spectrum.ieee.org/feeds/feed.rss"),
    ],
    "人才与组织": [
        ("Harvard Business Review", "https://hbr.org/feed"),
    ],
    "观点与言论（高信噪比）": [
        ("Project Syndicate", "https://www.project-syndicate.org/rss"),
    ],
    "机会雷达（Deal/合作/并购/融资）": [
        ("Crunchbase News (unofficial)", "https://news.crunchbase.com/feed/"),
    ],
}

ARXIV_LISTING: Dict[str, str] = {
    "论文雷达：机器学习（cs.LG）": "https://arxiv.org/list/cs.LG/recent",
    "论文雷达：经济学（econ.GN）": "https://arxiv.org/list/econ.GN/recent",
    "论文雷达：计算机与社会（cs.CY）": "https://arxiv.org/list/cs.CY/recent",
}

KEYWORDS: Dict[str, List[str]] = {
    "宏观与政策（全球）": ["央行","利率","通胀","就业","监管","财政","税","GDP","policy","inflation","rate","CPI","central bank"],
    "地缘政治与制裁合规": ["制裁","出口管制","关税","合规","OFAC","sanction","export control","tariff","entity list"],
    "AI 基础设施与算力链": ["GPU","推理","训练","数据中心","算力","NVIDIA","AMD","TPU","inference","datacenter","power","cooling"],
    "企业软件与搜索/RAG": ["RAG","检索","rerank","向量","embedding","权限","数据治理","企业搜索","evaluation","observability"],
    "跨境供应链与大宗/再生金属": ["铜","铝","杂铜","运价","海运","铁路","港口","基差","套保","期货","scrap","freight","port"],
    "企业采购与福利": ["预付卡","福利","企业购","结算","资金存管","发卡","MRO","procurement","benefits"],
    "金融市场与风险": ["信用","利差","债券","汇率","波动","bank","credit","yield","FX","risk"],
    "消费品牌与零售": ["DTC","零售","电商","渠道","定价","品牌","retail","ecommerce","pricing"],
    "材料与可持续": ["材料","回收","生物基","LCA","可持续","recycled","biobased","circular"],
    "碳市场与ESG": ["碳","碳信用","ESG","减排","披露","carbon","offset","VCM"],
    "硬件与IoT": ["传感器","摄像头","边缘","低功耗","BLE","Wi-Fi","IoT","sensor","edge"],
    "人才与组织": ["OKR","组织","管理","绩效","财务体系","预算","culture","leadership"],
    "观点与言论（高信噪比）": ["opinion","commentary","观点","专栏","speech","interview"],
    "机会雷达（Deal/合作/并购/融资）": ["融资","并购","收购","合作","招标","deal","acquisition","funding","tender"],
}

AUTHORITY_WEIGHTS: Dict[str, float] = {
    "imf.org": 2.0, "worldbank.org": 2.0, "oecd.org": 1.9,
    "un.org": 1.8, "treasury.gov": 1.8, "consilium.europa.eu": 1.7,
    "arxiv.org": 1.5,
    "spectrum.ieee.org": 1.4, "ieee.org": 1.4,
    "hbr.org": 1.2, "infoq.com": 1.2, "carbonbrief.org": 1.3,
    "techcrunch.com": 1.2, "theverge.com": 1.1,
    "investopedia.com": 1.1, "greenbiz.com": 1.1,
    "retaildive.com": 1.0, "project-syndicate.org": 1.1,
}

DEFAULT_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"

# -----------------------------
# 市场监控（汇率/利率/金属）
# -----------------------------
# 说明：Stooq 的 CSV 下载接口为：
#   https://stooq.com/q/d/l/?s=<symbol>&i=d
# 已验证的示例（仅作参考）：USDCNY、10YUSY.B、DX.F、HG.F、X8.F
# 你可随时在这里增删 watchlist。
MARKET_WATCHLIST = [
    # name, stooq_symbol, kind
    ("USD/CNY", "usdcny", "pct"),          # 货币：近30日百分比变化
    ("美国10Y国债收益率", "10yusy.b", "bp"),# 收益率：近30日变化（bp）
    ("美元指数DXY", "dx.f", "pct"),        # 美元指数期货：pct
    ("铜（COMEX）", "hg.f", "pct"),        # 铜期货：pct
    ("铝（LME溢价EU）", "x8.f", "pct"),    # 作为“铝”代理；如你找到更合适symbol可替换
]

@dataclass
class Item:
    section: str
    title: str
    link: str
    summary: str
    published: str
    published_ts: Optional[float]
    source: str
    domain: str
    tags: List[str]
    report_link: str = ""
    score: float = 0.0

def _domain_from_url(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""

def _strip_html(s: str) -> str:
    s = unescape(s or "")
    s = re.sub(r"<script.*?>.*?</script>", " ", s, flags=re.S|re.I)
    s = re.sub(r"<style.*?>.*?</style>", " ", s, flags=re.S|re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _parse_published_to_ts(published: str) -> Optional[float]:
    if not published:
        return None
    try:
        d = parsedate_to_datetime(published)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.timestamp()
    except Exception:
        pass
    try:
        d = dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.timestamp()
    except Exception:
        return None

def _truncate_or_compress(summary: str, limit: int = 500) -> str:
    s = _strip_html(summary)
    if len(s) <= limit:
        return s
    parts = re.split(r"[。！？；;\n]", s)
    acc = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(acc) + len(p) + 1 > limit - 10:
            break
        acc = (acc + "。") if acc else ""
        acc += p
    if len(acc) < 60:
        acc = s[: limit - 3]
    return acc[: limit - 1] + "…"

def _contains_numbers(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))

def _keyword_hits(text: str, keywords: List[str]) -> int:
    t = (text or "").lower()
    return sum(1 for kw in keywords if kw.lower() in t)

def _freshness(age_hours: float, window_hours: float) -> float:
    return max(0.0, 1.0 - (max(0.0, age_hours) / window_hours))

def _authority(domain: str) -> float:
    return AUTHORITY_WEIGHTS.get(domain, 1.0)

def _infer_report_link(link: str) -> str:
    if "arxiv.org/abs/" in link:
        return link.replace("/abs/", "/pdf/") + ".pdf"
    if link.lower().endswith(".pdf"):
        return link
    return ""


def _http_get(url: str, timeout: int = 12) -> bytes:
    """
    稳健抓取：直接抓取失败时，自动降级到代理抓取。
    目的：避免部分站点 403 / TLS / 区域限制导致“整栏为空”。
    说明：代理仅用于 RSS/CSV 等公开内容，不做登录态抓取。
    """
    def _one(u: str) -> bytes:
        req = urllib.request.Request(
            u,
            headers={
                "User-Agent": DEFAULT_UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Encoding": "gzip",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if "gzip" in (resp.headers.get("Content-Encoding", "").lower()):
                try:
                    data = gzip.decompress(data)
                except Exception:
                    pass
            return data

    # 1) 直连（重试）
    last_err: Exception | None = None
    for _ in range(2):
        try:
            return _one(url)
        except Exception as e:
            last_err = e

    # 2) allorigins raw 代理
    try:
        proxy = "https://api.allorigins.win/raw?url=" + urllib.parse.quote(url, safe="")
        return _one(proxy)
    except Exception as e:
        last_err = e

    # 3) jina ai reader（有时对部分站点更稳）
    try:
        # r.jina.ai 会把页面转为纯文本；对 RSS/CSV 一般也可用，但不保证
        if url.startswith("https://"):
            j = "https://r.jina.ai/http://" + url[len("https://"):]
        elif url.startswith("http://"):
            j = "https://r.jina.ai/http://" + url[len("http://"):]
        else:
            j = url
        return _one(j)
    except Exception as e:
        last_err = e

    raise last_err if last_err else RuntimeError("fetch_failed")


def fetch_rss(section: str, source_name: str, feed_url: str, timeout: int) -> List[Item]:
    out: List[Item] = []
    try:
        raw = _http_get(feed_url, timeout=timeout)
        root = ET.fromstring(raw)
        channel = root.find("channel")
        if channel is not None:
            for it in channel.findall("item"):
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                desc = (it.findtext("description") or it.findtext("summary") or "").strip()
                pub = (it.findtext("pubDate") or it.findtext("date") or "").strip()
                out.append(Item(
                    section=section,
                    title=title,
                    link=link,
                    summary=_truncate_or_compress(desc, 500),
                    published=pub,
                    published_ts=_parse_published_to_ts(pub),
                    source=source_name,
                    domain=_domain_from_url(link),
                    tags=[],
                ))
        else:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                link_el = entry.find("atom:link", ns)
                link = (link_el.get("href") if link_el is not None else "").strip()
                summ = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
                pub = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
                out.append(Item(
                    section=section,
                    title=title,
                    link=link,
                    summary=_truncate_or_compress(summ, 500),
                    published=pub,
                    published_ts=_parse_published_to_ts(pub),
                    source=source_name,
                    domain=_domain_from_url(link),
                    tags=[],
                ))
    except Exception:
        return []
    return out

def fetch_arxiv_recent(section: str, listing_url: str, timeout: int) -> List[Item]:
    out: List[Item] = []
    try:
        raw = _http_get(listing_url, timeout=timeout)
        html = raw.decode("utf-8", errors="ignore")
        pat = re.compile(r'arXiv:(\d{4}\.\d{5})[\s\S]{0,450}?Title:\s*(.+?)\s*(?:\n|<)', re.I)
        for m in pat.finditer(html):
            arxiv_id = m.group(1).strip()
            title = _strip_html(m.group(2).strip())
            link = f"https://arxiv.org/abs/{arxiv_id}"
            out.append(Item(
                section=section,
                title=title,
                link=link,
                summary="（建议点击查看摘要与PDF；此条来自arXiv recent列表，未抓取完整abstract。）",
                published="",
                published_ts=None,
                source="arXiv",
                domain="arxiv.org",
                tags=[],
            ))
    except Exception:
        return []
    return out

def fetch_stooq_series(symbol: str, days: int = 60, timeout: int = 10) -> Optional[List[Tuple[str, float]]]:
    # Stooq: q/d/l/?s=<sym>&i=d
    url = f"https://stooq.com/q/d/l/?s={urllib.parse.quote(symbol)}&i=d"
    try:
        raw = _http_get(url, timeout=timeout).decode("utf-8", errors="ignore")
        f = io.StringIO(raw)
        rows = list(csv.DictReader(f))
        if not rows:
            return None
        tail = rows[-days:]
        out = []
        for r in tail:
            d = r.get("Date")
            c = r.get("Close")
            if not d or not c or c == "null":
                continue
            try:
                out.append((d, float(c)))
            except Exception:
                pass
        return out if len(out) >= 10 else None
    except Exception:
        return None

def to_sparkline(points: List[float], width: int = 260, height: int = 56) -> str:
    if not points:
        return ""
    mn, mx = min(points), max(points)
    if mx - mn < 1e-9:
        mx = mn + 1.0
    pad = 4
    w = width - pad*2
    h = height - pad*2
    coords = []
    for i, v in enumerate(points):
        x = pad + (i/(len(points)-1))*w
        y = pad + (1 - (v - mn)/(mx - mn))*h
        coords.append(f"{x:.2f},{y:.2f}")
    poly = " ".join(coords)
    return f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <polyline fill="none" stroke="currentColor" stroke-width="2" points="{poly}" />
</svg>"""

def norm_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s*[-|–—]\s*(reuters|ap|bloomberg|cnn|bbc|ft|wsj)\s*$", "", t, flags=re.I)
    t = re.sub(r"[\W_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def dedup(items: List[Item], title_sim_threshold: float = 0.92) -> List[Item]:
    seen_url = set()
    tmp = []
    for it in items:
        if not it.link:
            continue
        if it.link in seen_url:
            continue
        seen_url.add(it.link)
        tmp.append(it)

    by_sec: Dict[str, List[Item]] = {}
    for it in tmp:
        by_sec.setdefault(it.section, []).append(it)

    out: List[Item] = []
    for sec, arr in by_sec.items():
        kept: List[Item] = []
        kept_norm: List[str] = []
        for it in arr:
            nt = norm_title(it.title)
            if not nt:
                kept.append(it); kept_norm.append(nt); continue
            is_dup = False
            for prev in kept_norm:
                if not prev:
                    continue
                r = difflib.SequenceMatcher(a=nt, b=prev).ratio()
                if r >= title_sim_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(it)
                kept_norm.append(nt)
        out.extend(kept)
    return out

def score_items(items: List[Item], now_ts: float, window_hours: float, balance: str) -> List[Item]:
    core_business = {
        "AI 基础设施与算力链",
        "企业软件与搜索/RAG",
        "跨境供应链与大宗/再生金属",
        "企业采购与福利",
        "材料与可持续",
        "碳市场与ESG",
        "硬件与IoT",
    }
    macro_geo = {"宏观与政策（全球）", "地缘政治与制裁合规", "金融市场与风险"}

    def section_weight(sec: str) -> float:
        if balance == "C":
            return 1.2 if (sec in core_business or sec in macro_geo) else 1.0
        if balance == "A":
            return 1.3 if sec in core_business else 1.0
        if balance == "B":
            return 1.3 if sec in macro_geo else 1.0
        return 1.0

    for it in items:
        text = f"{it.title} {it.summary}"
        kw = KEYWORDS.get(it.section, [])
        rel_hits = _keyword_hits(text, kw)
        rel = min(1.0, rel_hits / 3.0)

        if it.published_ts:
            age_h = (now_ts - it.published_ts) / 3600.0
            fresh = _freshness(age_h, window_hours)
        else:
            fresh = 0.60

        auth = _authority(it.domain) / 2.0
        auth = max(0.5, min(1.1, auth))

        density = 1.0 if _contains_numbers(text) else 0.0

        score = (2.0 * rel) + (1.6 * fresh) + (1.2 * auth) + (0.4 * density)
        score *= section_weight(it.section)
        it.score = score

        tags = []
        if it.section in macro_geo: tags.append("宏观/地缘")
        if it.section in core_business: tags.append("业务相关")
        if it.domain and it.domain != "example.com": tags.append(it.domain)
        if density: tags.append("含数字信号")
        it.tags = tags
        it.report_link = _infer_report_link(it.link)

    return items

def select_top10(items: List[Item], per_section_cap: int = 2, per_source_cap: int = 2) -> List[Item]:
    items_sorted = sorted(items, key=lambda x: x.score, reverse=True)
    sec_cnt: Dict[str, int] = {}
    src_cnt: Dict[str, int] = {}
    out: List[Item] = []
    for it in items_sorted:
        if len(out) >= 10:
            break
        if sec_cnt.get(it.section, 0) >= per_section_cap:
            continue
        if src_cnt.get(it.source, 0) >= per_source_cap:
            continue
        out.append(it)
        sec_cnt[it.section] = sec_cnt.get(it.section, 0) + 1
        src_cnt[it.source] = src_cnt.get(it.source, 0) + 1
    return out

def fmt_market_line(name: str, m: Dict) -> Optional[str]:
    if not m:
        return None
    kind = m.get("kind", "pct")
    if kind == "bp" and m.get("bp") is not None:
        direction = "上行" if m["bp"] >= 0 else "下行"
        return f"{name}近一月{direction}{abs(m['bp']):.1f}bp"
    if m.get("pct") is not None:
        direction = "上行" if m["pct"] >= 0 else "下行"
        return f"{name}近一月{direction}{abs(m['pct']):.2f}%"
    return None

def generate_daily_brief(top10: List[Item], markets: Dict[str, Dict]) -> str:
    by_sec: Dict[str, List[Item]] = {}
    for it in top10:
        by_sec.setdefault(it.section, []).append(it)

    def pick(sec: str) -> Optional[str]:
        arr = by_sec.get(sec) or []
        return arr[0].title if arr else None

    parts = []
    macro = pick("宏观与政策（全球）")
    risk = pick("金融市场与风险")
    geo  = pick("地缘政治与制裁合规")

    if macro or risk:
        s = "宏观与金融层面，"
        if macro:
            s += f"市场关注“{macro}”等信号，"
        if risk:
            s += f"同时“{risk}”提示风险偏好与资金价格可能出现再定价，"
        s += "建议把利率/通胀、美元强弱与信用利差作为今日核心变量。"
        parts.append(s)

    if geo:
        parts.append(f"地缘与制裁方面，“{geo}”意味着合规边界与供应链替代成本仍在上升，涉及跨境贸易与高性能计算等方向需更早做情景预案。")

    ai = pick("AI 基础设施与算力链")
    rag = pick("企业软件与搜索/RAG")
    sup = pick("跨境供应链与大宗/再生金属")
    ben = pick("企业采购与福利")

    if ai or rag:
        s = "在 AI 基建与企业软件上，"
        if ai:
            s += f"“{ai}”反映算力扩张正被电力/散热/供应约束牵引，"
        if rag:
            s += f"而“{rag}”指向企业侧更关注可评测、可治理、可落地的RAG闭环，"
        s += "优先跟踪成本曲线与工程化落地（评测、权限、数据治理）。"
        parts.append(s)

    if sup:
        parts.append(f"供应链与大宗（金属）方面，“{sup}”提示运价/港口/交期不确定性仍高，现货出售与套保/基差管理的组合策略价值上升。")
    if ben:
        parts.append(f"采购与福利合规方面，“{ben}”表明监管更看重资金存管与凭证链路，建议将制度、对账与留痕流程作为优先级。")

    mat = pick("材料与可持续")
    carb = pick("碳市场与ESG")
    if mat or carb:
        s = "可持续方向，"
        if mat:
            s += f"“{mat}”显示材料端的认证与成本仍是规模化门槛，"
        if carb:
            s += f"“{carb}”强调数据可追溯与方法学质量，"
        s += "对品牌与碳信用项目的交付应强化证据链与披露口径。"
        parts.append(s)

    if markets:
        lines = []
        for k, v in markets.items():
            line = fmt_market_line(k, v)
            if line:
                lines.append(line)
        if lines:
            parts.append("汇率/利率/金属走势方面，" + "，".join(lines) + "。")

    brief = " ".join(parts)
    brief = re.sub(r"\s+", " ", brief).strip()

    if len(brief) < 300:
        brief += " 今日阅读建议：先看 Top 10 把握全局变量，再按栏目进入与你业务最相关的两到三条深读；对可能触发“监管/制裁/运价异常”的信息，优先记录可执行的下一步动作与风险阈值。"
    if len(brief) > 500:
        brief = brief[:497] + "…"
    return brief

def demo_items() -> List[Item]:
    now = dt.datetime.now(dt.timezone.utc).timestamp()
    def mk(section, title, link, source, hours_ago, summary):
        ts = now - hours_ago * 3600
        pub = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        return Item(section, title, link, _truncate_or_compress(summary, 500), pub, ts, source, _domain_from_url(link), [])
    return [
        mk("企业采购与福利","预付卡监管趋严：资金存管与冲抵凭证成为审查重点","https://example.com/benefit1","Finextra",9,"关注点：合规流程、账户结构、票据与对账链路。"),
        mk("地缘政治与制裁合规","欧盟更新出口管制清单，涉及高性能计算部件","https://example.com/sanction1","EU Council",10,"关注点：合规范围扩大、转口风险、供应链替代与审计要求。"),
        mk("硬件与IoT","低功耗传感与边缘计算结合，提升连续监测场景可靠性","https://example.com/iot1","IEEE Spectrum",15,"关注点：功耗预算、传输策略、隐私与合规。"),
        mk("碳市场与ESG","自愿减排方法学更新：更强调额外性与数据可追溯","https://example.com/carbon1","Carbon Brief",18,"关注点：项目质量、登记体系、披露规则。"),
        mk("AI 基础设施与算力链","数据中心电力瓶颈成为AI扩张主要约束，液冷加速渗透","https://example.com/aiinfra1","The Verge",8,"关注点：PUE、液冷渗透率、供电审批周期。"),
        mk("宏观与政策（全球）","央行暗示维持利率更久，市场重估通胀路径","https://example.com/macro1","IMF",6,"关注点：通胀粘性、财政扩张与就业韧性之间的再平衡。"),
        mk("金融市场与风险","信用利差扩大与再融资窗口收紧，关注高杠杆企业违约","https://example.com/risk1","Investopedia",7,"关注点：利差、到期墙、银行风险偏好。"),
        mk("企业软件与搜索/RAG","企业级RAG评测：从离线指标走向在线任务成功率","https://example.com/rag1","InfoQ",14,"关注点：多语言、权限过滤、rerank与人机反馈闭环。"),
        mk("跨境供应链与大宗/再生金属","运价波动与港口拥堵导致交期不确定性上升，建议分段对冲","https://example.com/supply1","Lloyds",12,"关注点：基差与套保、仓储与现货出售策略。"),
        mk("材料与可持续","生物基材料在耐久与成本之间的折中，认证成为门槛","https://example.com/material1","GreenBiz",20,"关注点：LCA、认证、规模化产能。"),
    ]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>今日资讯简报 - __DATE__</title>
  <style>
    :root { --bg:#f2f2f7; --card:#fff; --text:#1c1c1e; --muted:#6e6e73; --line:rgba(60,60,67,.12); --shadow:0 6px 18px rgba(0,0,0,.06); --blue:#0a84ff; }
    *{box-sizing:border-box}
    body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Helvetica Neue","PingFang SC","Hiragino Sans GB","Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--text)}
    .layout{display:grid;grid-template-columns:320px 1fr;min-height:100vh}
    .sidebar{position:sticky;top:0;height:100vh;padding:18px;border-right:1px solid var(--line);background:linear-gradient(180deg,rgba(255,255,255,.70),rgba(255,255,255,.35));backdrop-filter:blur(14px);display:flex;flex-direction:column;gap:14px}
    .brand{background:var(--card);border-radius:18px;box-shadow:var(--shadow);padding:16px 16px 14px}
    .brand h1{margin:0 0 6px;font-size:20px;font-weight:700;letter-spacing:-.2px}
    .brand .date{color:var(--muted);font-size:13px}
    .search{margin-top:12px;display:flex;gap:8px}
    .search input{flex:1;padding:10px 12px;border-radius:12px;border:1px solid var(--line);background:rgba(255,255,255,.85);outline:none;font-size:14px}
    .nav{background:var(--card);border-radius:18px;box-shadow:var(--shadow);padding:10px;overflow-y:auto;-webkit-overflow-scrolling:touch;flex:1;min-height:0}
    .nav .group-title{padding:10px 12px 6px;font-size:13px;color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--card)}
    .nav button{width:100%;text-align:left;padding:10px 12px;border:0;background:transparent;border-radius:12px;cursor:pointer;font-size:14px;color:var(--text)}
    .nav button:hover{background:rgba(10,132,255,.08)}
    .nav button.active{background:rgba(10,132,255,.12);color:#0040dd;font-weight:700}
    .main{padding:22px 24px 60px;max-width:1180px}
    .hero{background:linear-gradient(180deg,rgba(10,132,255,.18),rgba(10,132,255,.04));border:1px solid rgba(10,132,255,.18);border-radius:22px;padding:18px 20px;box-shadow:0 10px 24px rgba(10,132,255,.08);margin-bottom:16px}
    .hero h2{margin:0 0 6px;font-size:28px;letter-spacing:-.3px}
    .hero .subtitle{color:var(--muted);font-size:14px;line-height:1.5}
    .toolbar{display:flex;align-items:center;justify-content:space-between;margin:14px 0 10px}
    .toolbar .title{font-size:22px;font-weight:800;letter-spacing:-.2px}
    .pill{border:1px solid var(--line);background:rgba(255,255,255,.8);padding:8px 12px;border-radius:999px;color:var(--muted);font-size:13px}
    .brief{background:var(--card);border-radius:18px;box-shadow:var(--shadow);padding:14px 16px;border:1px solid rgba(60,60,67,.10);margin-bottom:12px}
    .brief h3{margin:0 0 8px;font-size:16px;font-weight:800}
    .brief p{margin:0;color:#2c2c2e;font-size:14px;line-height:1.6}
    .markets{margin-top:10px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
    .mcard{border:1px solid var(--line);border-radius:14px;padding:10px 12px;background:rgba(255,255,255,.75)}
    .mhead{display:flex;justify-content:space-between;align-items:baseline;color:var(--muted);font-size:12px;margin-bottom:6px}
    .mname{font-weight:800;color:var(--text)}
    .mstat{font-weight:700}
    .spark{color:var(--blue)}
    .list{display:grid;grid-template-columns:1fr;gap:12px}
    .card{background:var(--card);border-radius:18px;box-shadow:var(--shadow);padding:14px 16px;border:1px solid rgba(60,60,67,.10)}
    .card .topline{display:flex;gap:10px;align-items:baseline;justify-content:space-between;margin-bottom:6px}
    .card a.title{color:var(--text);font-weight:800;text-decoration:none;font-size:16px;line-height:1.3}
    .card a.title:hover{color:var(--blue)}
    .meta{color:var(--muted);font-size:12px;white-space:nowrap}
    .summary{margin-top:6px;color:#2c2c2e;font-size:14px;line-height:1.55}
    .tags{margin-top:10px;display:flex;flex-wrap:wrap;gap:8px}
    .tag{font-size:12px;color:#3a3a3c;border:1px solid var(--line);background:rgba(255,255,255,.7);padding:4px 10px;border-radius:999px}
    .actions{margin-top:10px;display:flex;gap:10px;flex-wrap:wrap}
    .actions a{font-size:13px;text-decoration:none;color:var(--blue);font-weight:700}
    .actions a.secondary{color:#5e5ce6}
    .empty{color:var(--muted);font-size:14px;padding:10px 2px}
    @media (max-width:960px){
      .layout{grid-template-columns:1fr}
      .sidebar{position:relative;height:auto}
      .nav{max-height:52vh}
      .main{padding:18px}
      .markets{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">
      <h1>今日资讯简报</h1>
      <div class="date">日期：__DATE__</div>
      <div class="search"><input id="q" placeholder="搜索标题/摘要/标签…" /></div>
    </div>
    <div class="nav" id="nav"></div>
  </aside>

  <main class="main">
    <section class="hero">
      <h2>把碎片推送收束为一页</h2>
      <div class="subtitle">覆盖：宏观与政策 / 地缘与制裁 / AI 基建 / 企业搜索 / 供应链与大宗 / 采购与福利 / 金融风险 / 消费零售 / 材料与可持续 / 碳市场 / IoT / 人才组织 / 观点言论 / 机会雷达 / 异常预警 / 论文雷达。并增加：汇率 / 利率 / 金属趋势。</div>
    </section>

    <div class="toolbar">
      <div class="title" id="sectionTitle">Top 10 今日要点</div>
      <div class="pill" id="sectionHint">点击左侧栏目切换；输入搜索过滤</div>
    </div>

    <div id="briefMount"></div>
    <div class="list" id="list"></div>
  </main>
</div>

<script>
const DIGEST = __DATA__;
let state = { section: "Top 10 今日要点", query: "" };

function renderNav(){
  const nav = document.getElementById("nav");
  nav.innerHTML = "";
  const gt = document.createElement("div");
  gt.className = "group-title";
  gt.textContent = "导航";
  nav.appendChild(gt);

  for (const name of DIGEST.section_order){
    const btn = document.createElement("button");
    btn.setAttribute("data-section", name);
    btn.className = (name===state.section ? "active" : "");
    btn.textContent = name;
    btn.addEventListener("click", () => { state.section = name; state.query = document.getElementById("q").value || ""; render(); });
    nav.appendChild(btn);
  }
}

function filterItems(items){
  const q = (state.query || "").trim().toLowerCase();
  if (!q) return items;
  return items.filter(it => (((it.title||"") + " " + (it.summary||"") + " " + ((it.tags||[]).join(" "))).toLowerCase().includes(q)));
}

function formatMarketStat(m){
  if (!m) return "暂无数据";
  if (m.kind === "bp" && typeof m.bp === "number"){
    const sign = m.bp >= 0 ? "+" : "-";
    return `${sign}${Math.abs(m.bp).toFixed(1)}bp（近30日）`;
  }
  if (typeof m.pct === "number"){
    const sign = m.pct >= 0 ? "+" : "-";
    return `${sign}${Math.abs(m.pct).toFixed(2)}%（近30日）`;
  }
  return "暂无数据";
}

function renderBriefIfTop10(){
  const mount = document.getElementById("briefMount");
  mount.innerHTML = "";
  if (state.section !== "Top 10 今日要点") return;

  const brief = document.createElement("div");
  brief.className = "brief";

  const h3 = document.createElement("h3");
  h3.textContent = "今日简报（300–500字）";
  brief.appendChild(h3);

  const p = document.createElement("p");
  p.textContent = DIGEST.daily_brief || "";
  brief.appendChild(p);

  const markets = DIGEST.markets || {};
  const keys = Object.keys(markets);
  if (keys.length){
    const grid = document.createElement("div");
    grid.className = "markets";

    for (const k of keys){
      const m = markets[k] || {};
      const card = document.createElement("div");
      card.className = "mcard";

      const head = document.createElement("div");
      head.className = "mhead";

      const name = document.createElement("div");
      name.className = "mname";
      name.textContent = k;

      const stat = document.createElement("div");
      stat.className = "mstat";
      stat.textContent = formatMarketStat(m);

      head.appendChild(name);
      head.appendChild(stat);

      const spark = document.createElement("div");
      spark.className = "spark";
      spark.innerHTML = m.sparkline_svg || "";

      card.appendChild(head);
      card.appendChild(spark);
      grid.appendChild(card);
    }
    brief.appendChild(grid);
  }
  mount.appendChild(brief);
}

function renderList(){
  const list = document.getElementById("list");
  list.innerHTML = "";

  let items = [];
  if (state.section === "Top 10 今日要点") items = DIGEST.top10 || [];
  else items = (DIGEST.sections[state.section] || []);
  items = filterItems(items);

  if (!items.length){
    const e = document.createElement("div");
    e.className = "empty";
    e.textContent = "当前栏目暂无可显示条目（或被搜索过滤）。";
    list.appendChild(e);
    return;
  }

  for (const it of items){
    const card = document.createElement("div");
    card.className = "card";

    const topline = document.createElement("div");
    topline.className = "topline";

    const titleA = document.createElement("a");
    titleA.className = "title";
    titleA.href = it.link || "#";
    titleA.target = "_blank";
    titleA.textContent = it.title || "(无标题)";

    const meta = document.createElement("div");
    meta.className = "meta";
    const pub = it.published ? it.published : "";
    meta.textContent = (it.source||"") + (pub ? " · " + pub : "") + (it.score ? " · score " + Number(it.score).toFixed(2) : "");

    topline.appendChild(titleA);
    topline.appendChild(meta);

    const summary = document.createElement("div");
    summary.className = "summary";
    summary.textContent = it.summary || "";

    const tags = document.createElement("div");
    tags.className = "tags";
    for (const t of (it.tags || [])){
      const tg = document.createElement("span");
      tg.className = "tag";
      tg.textContent = t;
      tags.appendChild(tg);
    }

    const actions = document.createElement("div");
    actions.className = "actions";
    const a1 = document.createElement("a");
    a1.href = it.link || "#";
    a1.target = "_blank";
    a1.textContent = "打开原文";
    actions.appendChild(a1);

    if (it.report_link){
      const a2 = document.createElement("a");
      a2.href = it.report_link;
      a2.target = "_blank";
      a2.className = "secondary";
      a2.textContent = "打开报告/PDF";
      actions.appendChild(a2);
    }

    card.appendChild(topline);
    card.appendChild(summary);
    if ((it.tags || []).length) card.appendChild(tags);
    card.appendChild(actions);

    list.appendChild(card);
  }
}

function render(){
  document.getElementById("sectionTitle").textContent = state.section;
  document.querySelectorAll(".nav button").forEach(b => b.classList.toggle("active", b.getAttribute("data-section") === state.section));
  renderBriefIfTop10();
  renderList();
}

document.getElementById("q").addEventListener("input", (e) => { state.query = e.target.value || ""; renderList(); });

renderNav();
render();
</script>
</body>
</html>
"""

def build_markets(timeout: int, skip_markets: bool, demo: bool) -> Dict[str, Dict]:
    markets: Dict[str, Dict] = {}
    if skip_markets:
        return markets

    if demo:
        # 仅用于演示布局：构造几条假数据
        def mk(points):
            return {"sparkline_svg": to_sparkline(points)}
        seq1 = [7.10 + 0.001*i + math.sin(i/4)*0.01 for i in range(30)]
        seq2 = [4.00 + 0.01*i + math.sin(i/3)*0.04 for i in range(30)]
        seq3 = [103 + math.sin(i/2)*1.6 for i in range(30)]
        seq4 = [520 + 2*i + math.sin(i/2)*12 for i in range(30)]
        seq5 = [250 + 1.2*i + math.sin(i/2)*9 for i in range(30)]
        markets["USD/CNY"] = {"symbol":"usdcny","kind":"pct","pct":(seq1[-1]/seq1[0]-1)*100, **mk(seq1)}
        markets["美国10Y国债收益率"] = {"symbol":"10yusy.b","kind":"bp","bp":(seq2[-1]-seq2[0])*100, **mk(seq2)}
        markets["美元指数DXY"] = {"symbol":"dx.f","kind":"pct","pct":(seq3[-1]/seq3[0]-1)*100, **mk(seq3)}
        markets["铜（COMEX）"] = {"symbol":"hg.f","kind":"pct","pct":(seq4[-1]/seq4[0]-1)*100, **mk(seq4)}
        markets["铝（LME溢价EU）"] = {"symbol":"x8.f","kind":"pct","pct":(seq5[-1]/seq5[0]-1)*100, **mk(seq5)}
        return markets

    for name, sym, kind in MARKET_WATCHLIST:
        series = fetch_stooq_series(sym, days=60, timeout=timeout)
        if not series:
            continue
        closes = [x[1] for x in series][-30:]
        if len(closes) < 2:
            continue

        if kind == "bp":
            # 10Y收益率：显示近30日变化（bp）
            bp = (closes[-1] - closes[0]) * 100.0  # 1.00% = 100bp
            markets[name] = {
                "symbol": sym, "kind": "bp",
                "bp": bp,
                "sparkline_svg": to_sparkline(closes)
            }
        else:
            pct = (closes[-1] / closes[0] - 1.0) * 100.0
            markets[name] = {
                "symbol": sym, "kind": "pct",
                "pct": pct,
                "sparkline_svg": to_sparkline(closes)
            }
    return markets

def build_digest_json(items: List[Item], section_order: List[str], window_hours: float, balance: str, markets: Dict[str, Dict]) -> Dict:
    now_ts = dt.datetime.now(dt.timezone.utc).timestamp()

    filtered: List[Item] = []
    for it in items:
        if it.published_ts is None:
            filtered.append(it)
            continue
        age_h = (now_ts - it.published_ts) / 3600.0
        if age_h <= window_hours:
            filtered.append(it)

    filtered = dedup(filtered, title_sim_threshold=0.92)
    filtered = score_items(filtered, now_ts=now_ts, window_hours=window_hours, balance=balance)

    sections: Dict[str, List[Item]] = {}
    for it in filtered:
        sections.setdefault(it.section, []).append(it)
    for sec in sections:
        sections[sec].sort(key=lambda x: x.score, reverse=True)

    top_pool = [it for it in filtered if not it.section.startswith("论文雷达") and it.section != "异常与预警（触发才显示）"]
    top10 = select_top10(top_pool, per_section_cap=2, per_source_cap=2)
    daily_brief = generate_daily_brief(top10, markets)

    def to_dict(it: Item) -> Dict:
        d = asdict(it)
        d.pop("published_ts", None)
        d["score"] = float(d.get("score") or 0.0)
        return d

    return {
        "date": dt.datetime.now().strftime("%Y年%m月%d日"),
        "balance": balance,
        "window_hours": window_hours,
        "daily_brief": daily_brief,
        "markets": markets,
        "section_order": section_order,
        "top10": [to_dict(x) for x in top10],
        "sections": {k: [to_dict(x) for x in v] for k, v in sections.items()},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="daily_digest.html")
    ap.add_argument("--max-per-section", type=int, default=12)
    ap.add_argument("--window-hours", type=float, default=36.0)
    ap.add_argument("--timeout", type=int, default=12)
    ap.add_argument("--skip-rss", action="store_true")
    ap.add_argument("--skip-arxiv", action="store_true")
    ap.add_argument("--skip-markets", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--balance", default="C", choices=["A","B","C"])
    args = ap.parse_args()

    items: List[Item] = []
    markets = build_markets(timeout=args.timeout, skip_markets=args.skip_markets, demo=args.demo)

    if args.demo:
        items = demo_items()
    else:
        if not args.skip_rss:
            for sec, feeds in RSS_FEEDS.items():
                for source, url in feeds:
                    items.extend(fetch_rss(sec, source, url, timeout=args.timeout))
        if not args.skip_arxiv:
            for sec, url in ARXIV_LISTING.items():
                items.extend(fetch_arxiv_recent(sec, url, timeout=args.timeout))

        by_sec: Dict[str, List[Item]] = {}
        for it in items:
            by_sec.setdefault(it.section, []).append(it)
        trimmed: List[Item] = []
        for sec, arr in by_sec.items():
            trimmed.extend(arr[: max(20, args.max_per_section * 3)])
        items = trimmed

    section_order = [s for s in SECTIONS_ORDER if (s == "Top 10 今日要点") or (s in RSS_FEEDS) or (s in ARXIV_LISTING)]
    digest = build_digest_json(items=items, section_order=section_order, window_hours=args.window_hours, balance=args.balance, markets=markets)

    html = HTML_TEMPLATE.replace("__DATE__", digest["date"]).replace("__DATA__", json.dumps(digest, ensure_ascii=False))
    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"OK: {out_path.resolve()}")

if __name__ == "__main__":
    main()
